from ops import *
from utils import *
from glob import glob
import time

class pix2pix(object):
    def __init__(self, sess, args):
        self.model_name = 'pix2pix'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.dataset_name = args.dataset

        self.epoch = args.epoch # 100000
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq

        self.ch = args.ch
        self.repeat = args.repeat

        """ Weight """
        self.L1_weight = args.L1_weight
        self.lr = args.lr

        self.img_size = args.img_size
        self.gray_to_RGB = args.gray_to_RGB

        if self.gray_to_RGB :
            self.input_ch = 1
            self.output_ch = 3
        else :
            self.input_ch = 3
            self.output_ch = 3

        self.trainA, self.trainB = prepare_data(dataset_name=self.dataset_name, size=self.img_size, gray_to_RGB=self.gray_to_RGB)
        self.num_batches = max(len(self.trainA), len(self.trainB)) // self.batch_size

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

    def generator(self, x, is_training=True, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x, channel, kernel=7, stride=1, pad=3, scope='conv_0')
            x = batch_norm(x, is_training, scope='conv_batch_0')
            x = relu(x)

            # Encoder
            for i in range(2) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, scope='en_conv_'+str(i))
                x = batch_norm(x, is_training, scope='en_batch_'+str(i))
                x = relu(x)
                channel = channel * 2

            # Bottle-neck
            for i in range(self.repeat) :
                x = resblock(x, channel, scope='resblock_'+str(i))

            # Decoder
            for i in range(2) :
                x = deconv(x, channel//2, kernel=4, stride=2, scope='deconv_'+str(i))
                x = batch_norm(x, is_training, scope='de_batch_'+str(i))
                x = relu(x)

                channel = channel // 2

            x = conv(x, channels=3, kernel=7, stride=1, pad=3, scope='last_conv') # NO BATCH NORM
            x = tanh(x)

            return x

    def discriminator(self, x, is_training=True, reuse=False, scope="discriminator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse):
            x = conv(x, channel, kernel=4, stride=2, pad=1, scope='first_conv') # NO BATCH NORM
            x = lrelu(x, 0.2)

            for i in range(2) :
                x = conv(x, channel*2, kernel=4, stride=2, pad=1, scope='conv_'+str(i))
                x = batch_norm(x, is_training, scope='batch_'+str(i))
                x = lrelu(x, 0.2)
                channel = channel * 2

            x = conv(x, channel, kernel=3, stride=1, pad=1, scope='conv_2')
            x = batch_norm(x, is_training, scope='batch_2')
            x = lrelu(x, 0.2)

            x = conv(x, channels=1, kernel=3, stride=1, pad=1, scope='last_conv')

            return x


    def build_model(self):

        """ Graph Image"""
        self.real_A = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.input_ch], name='real_A') # gray

        self.real_B = tf.placeholder(tf.float32, [self.batch_size, self.img_size, self.img_size, self.output_ch], name='real_B') # rgb


        """ Loss Function """
        D_real_logit = self.discriminator(self.real_B, reuse=False)

        self.fake_B = self.generator(self.real_A)
        D_fake_logit = self.discriminator(self.fake_B, reuse=True)

        self.d_loss = discriminator_loss(real=D_real_logit, fake=D_fake_logit)

        self.g_loss = generator_loss(fake=D_fake_logit) + self.L1_weight * L1_loss(self.real_B, self.fake_B)

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.g_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5, beta2=0.999).minimize(self.d_loss, var_list=D_vars)

        """" Summary """
        self.G_loss_summary = tf.summary.scalar("Generator_loss", self.g_loss)
        self.D_loss_summary = tf.summary.scalar("Discriminator_loss", self.d_loss)

        """ Test """
        self.test_real_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.input_ch], name='test_real_A')
        self.sample = self.generator(self.test_real_A, is_training=False, reuse=True)


    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_dir, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            self.trainA, self.trainB = shuffle(self.trainA, self.trainB)
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_A_images = self.trainA[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_B_images = self.trainB[idx * self.batch_size : (idx + 1) * self.batch_size]

                train_feed_dict = {
                    self.real_A : batch_A_images,
                    self.real_B : batch_B_images,
                }

                # Update D
                _, d_loss, summary_str = self.sess.run([self.D_optim, self.d_loss, self.D_loss_summary],
                                                       feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # Update G
                fake_B, _, g_loss, summary_str = self.sess.run([self.fake_B, self.G_optim, self.g_loss, self.G_loss_summary],
                                                               feed_dict=train_feed_dict)
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                if np.mod(counter, self.print_freq) == 0:
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:3d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx))
                    save_images(batch_B_images, [self.batch_size, 1],
                                './{}/real_B_{:03d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx))

                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:03d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx))

                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                # self.save(self.checkpoint_dir, counter)

                # save model for final step
            self.save(self.checkpoint_dir, counter)


    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, size=self.img_size, gray_to_RGB=self.gray_to_RGB))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.sample, feed_dict = {self.test_real_A : sample_image})

            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file), self.img_size, self.img_size))
            index.write("<td><img src='%s' width='%d' height='%d'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path), self.img_size, self.img_size))
            index.write("</tr>")

        index.close()
