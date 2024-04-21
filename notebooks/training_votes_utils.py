import os
import sys 

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # for compatibility with the rest of the project
import tensorflow_probability as tfp

def build_input_pipeline(data_dir):
  """Load data and build iterator for minibatches.
  
  Args:
    data_dir: The directory where the data is located. There must be four
      files inside the rep: `votes.npy`, `bill_indices.npy`, 
      `senator_indices.npy`, and `senator_map.txt`. `votes_npy` is a binary
      vector with shape [num_total_votes], indicating the outcome of each cast
      vote. `bill_indices.npy` is a vector with shape [num_total_votes], where
      each entry is an integer in {0, 1, ..., num_bills - 1}, indicating the
      bill index voted on. `senator_indices.npy` is a vector with shape 
      [num_total_votes], where each entry is an integer in {0, 1, ..., 
      num_senators - 1}, indicating the Senator voting. Finally, 
      `senator_map.txt` is a list of each Senator's name.
  """
  votes = np.load(os.path.join(data_dir, "votes.npy"))
  bill_indices = np.load(os.path.join(data_dir, "bill_indices.npy"))
  senator_indices = np.load(os.path.join(data_dir, "senator_indices.npy"))
  senator_map = np.loadtxt(os.path.join(data_dir, "senator_map.txt"),
                           dtype=str, 
                           delimiter="\n")
  num_bills = len(np.unique(bill_indices))
  num_senators = len(senator_map)
  dataset_size = len(votes)
  dataset = tf.data.Dataset.from_tensor_slices(
      (votes, bill_indices, senator_indices))
  # Use the complete dataset as a batch.
  batch_size = len(votes)
  batches = dataset.repeat().batch(batch_size).prefetch(batch_size)
  iterator = tf.compat.v1.data.make_one_shot_iterator(batches) # changed for compatibility issues
  return iterator, senator_map, num_bills, num_senators, dataset_size


def print_ideal_points(ideal_point_loc, senator_map):
  """Order and print ideal points for Tensorboard."""
  return ", ".join(senator_map[np.argsort(ideal_point_loc)])


def get_log_prior(samples):
  """Return log prior of sampled Gaussians.
  
  Args:
    samples: A Tensor with shape [num_samples, :].
  
  Returns:
    log_prior: A `Tensor` with shape [num_samples], with the log prior
      summed across the latent dimension.
  """
  prior_distribution = tfp.distributions.Normal(loc=0., scale=1.)
  log_prior = tf.reduce_sum(prior_distribution.log_prob(samples), axis=1)
  return log_prior


def get_entropy(distribution, samples):
  """Return entropy of sampled Gaussians.
  
  Args:
    samples: A Tensor with shape [num_samples, :].
  
  Returns:
    entropy: A `Tensor` with shape [num_samples], with the entropy
      summed across the latent dimension.
  """
  entropy = -tf.reduce_sum(distribution.log_prob(samples), axis=1)
  return entropy


def get_elbo(votes,
             bill_indices,
             senator_indices,
             ideal_point_distribution,
             polarity_distribution,
             popularity_distribution,
             dataset_size,
             num_samples):
  """Approximate ELBO using reparameterization.
  
  Args:
    votes: A binary vector with shape [batch_size].
    bill_indices: An int-vector with shape [batch_size].
    senator_indices: An int-vector with shape [batch_size].
    ideal_point_distribution: A Distribution object with parameter shape 
      [num_senators].
    polarity_distribution: A Distribution object with parameter shape 
      [num_bills].
    popularity_distribution: A Distribution object with parameter shape 
      [num_bills]
    dataset_size: The number of observations in the total data set (used to 
      calculate log-likelihood scale).
    num_samples: Number of Monte-Carlo samples.
  
  Returns:
    elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
      averaged across samples and summed across batches.
  """
  ideal_point_samples = ideal_point_distribution.sample(num_samples)
  polarity_samples = polarity_distribution.sample(num_samples)
  popularity_samples = popularity_distribution.sample(num_samples)
  
  ideal_point_log_prior = get_log_prior(ideal_point_samples)
  polarity_log_prior = get_log_prior(polarity_samples)
  popularity_log_prior = get_log_prior(popularity_samples)
  log_prior = ideal_point_log_prior + polarity_log_prior + popularity_log_prior

  ideal_point_entropy = get_entropy(ideal_point_distribution, 
                                    ideal_point_samples)
  polarity_entropy = get_entropy(polarity_distribution, polarity_samples)
  popularity_entropy = get_entropy(popularity_distribution, popularity_samples)
  entropy = ideal_point_entropy + polarity_entropy + popularity_entropy

  selected_ideal_points = tf.gather(ideal_point_samples, 
                                    senator_indices, 
                                    axis=1)
  selected_polarities = tf.gather(polarity_samples, bill_indices, axis=1) 
  selected_popularities = tf.gather(popularity_samples, bill_indices, axis=1) 
  vote_logits = (selected_ideal_points * 
                 selected_polarities + 
                 selected_popularities)

  vote_distribution = tfp.distributions.Bernoulli(logits=vote_logits)
  vote_log_likelihood = vote_distribution.log_prob(votes)
  vote_log_likelihood = tf.reduce_sum(vote_log_likelihood, axis=1)
  
  elbo = log_prior + vote_log_likelihood + entropy
  elbo = tf.reduce_mean(elbo)

  tf.summary.scalar("elbo/elbo", elbo)
  tf.summary.scalar("elbo/log_prior", tf.reduce_mean(log_prior))
  tf.summary.scalar("elbo/vote_log_likelihood", 
                    tf.reduce_mean(vote_log_likelihood))
  tf.summary.scalar("elbo/entropy", tf.reduce_mean(entropy))
  return elbo