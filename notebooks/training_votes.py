"""Learn vote ideal points for House of Commons using methods described by Vafa et al.

To preprocess data, make sure to run `preprocess_votes.py` before 
running.

### References
[1] Vafa, Keyon, Suresh Naidu, and David M. Blei. Text-Based Ideal Points, (2020). 
    https://github.com/keyonvafa/tbip/tree/master
"""

import os
import sys 

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 

data_dir = os.path.join(project_dir, "data/votes/input_2019")
save_dir = os.path.join(project_dir, "data/votes/output_2019")

import functools
import time

import training_votes_utils as utils

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # for compatibility with the rest of the project
import tensorflow_probability as tfp

#%%

# HYPERPARAMETERS

# adam learning rate
learning_rate = 0.01
# number of training steps to run
max_steps = 2500
# number of samples to use for ELBO approximation
num_samples = 1
# number of steps to print and save results
print_steps = 100
# random seed to be used
seed = 100

tf.set_random_seed(seed)

#%%

if tf.gfile.Exists(save_dir):
  tf.logging.warn("Deleting old log directory at {}".format(save_dir))
  tf.gfile.DeleteRecursively(save_dir)
tf.gfile.MakeDirs(save_dir)

(iterator, senator_map, num_bills, num_senators, dataset_size) = utils.build_input_pipeline(data_dir)
votes, bill_indices, senator_indices = iterator.get_next()

# initialize variational parameters.
ideal_point_loc = tf.get_variable("ideal_point_loc",
                                  shape=[num_senators],
                                  dtype=tf.float32)
ideal_point_logit = tf.get_variable("ideal_point_logit",
                                    shape=[num_senators],
                                    dtype=tf.float32)
polarity_loc = tf.get_variable("polarity_loc",
                               shape=[num_bills],
                               dtype=tf.float32)
polarity_logit = tf.get_variable("polarity_logit",
                                 shape=[num_bills],
                                 dtype=tf.float32)
popularity_loc = tf.get_variable("popularity_loc",
                                 shape=[num_bills],
                                 dtype=tf.float32)
popularity_logit = tf.get_variable("popularity_logit",
                                   shape=[num_bills],
                                   dtype=tf.float32)

ideal_point_scale = tf.nn.softplus(ideal_point_logit)
polarity_scale = tf.nn.softplus(polarity_logit)
popularity_scale = tf.nn.softplus(popularity_logit)

tf.summary.histogram("params/ideal_point_loc", ideal_point_loc)
tf.summary.histogram("params/ideal_point_scale", ideal_point_scale)
tf.summary.histogram("params/polarity_loc", polarity_loc)
tf.summary.histogram("params/polarity_scale", polarity_scale)
tf.summary.histogram("params/popularity_loc", popularity_loc)
tf.summary.histogram("params/popularity_scale", popularity_scale)

ideal_point_distribution = tfp.distributions.Normal(loc=ideal_point_loc,
                                                    scale=ideal_point_scale)
polarity_distribution = tfp.distributions.Normal(loc=polarity_loc,
                                                 scale=polarity_scale) 
popularity_distribution = tfp.distributions.Normal(loc=popularity_loc,
                                                   scale=popularity_scale)  

elbo = utils.get_elbo(votes,
                      bill_indices,
                      senator_indices,
                      ideal_point_distribution,
                      polarity_distribution,
                      popularity_distribution,
                      dataset_size,
                      num_samples)
loss = -elbo
tf.summary.scalar("loss", loss)

optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optim.minimize(loss)

ideal_point_list = tf.py_func(
    functools.partial(utils.print_ideal_points, senator_map=senator_map),
    [ideal_point_loc],
    tf.string, 
    stateful=False)
tf.summary.text("ideal_points", ideal_point_list) 

summary = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
  summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
  sess.run(init)
  for step in range(max_steps):
    start_time = time.time()
    (_, elbo_val) = sess.run([train_op, elbo])
    duration = time.time() - start_time
    if step % print_steps == 0:
      print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
          step, elbo_val, duration))
                   
      summary_str = sess.run(summary)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()
    
    if step % 100 == 0 or step == max_steps - 1:
      (ideal_point_loc_val, ideal_point_scale_val, polarity_loc_val,
       polarity_scale_val, popularity_loc_val, 
       popularity_scale_val) = sess.run([
           ideal_point_loc, ideal_point_scale, polarity_loc, 
           polarity_scale, popularity_loc, popularity_scale])
      if not tf.gfile.Exists(save_dir):
        tf.gfile.MakeDirs(save_dir)
      np.save(os.path.join(save_dir, "ideal_point_loc"), 
              ideal_point_loc_val)
      np.save(os.path.join(save_dir, "ideal_point_scale"), 
              ideal_point_scale_val)
      np.save(os.path.join(save_dir, "polarity_loc"), 
              polarity_loc_val)
      np.save(os.path.join(save_dir, "polarity_scale"), 
              polarity_scale_val)
      np.save(os.path.join(save_dir, "popularity_loc"), 
              popularity_loc_val)
      np.save(os.path.join(save_dir, "popularity_scale"), 
              popularity_scale_val)
