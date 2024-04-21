import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import pandas as pd
import numpy as npnano
import numpyro.distributions as dist
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import NMF
from optax import adam, exponential_decay
from numpyro.infer import SVI, TraceMeanField_ELBO
from jax import jit
from jax import random
from scipy import sparse
from numpyro import plate, sample, param
from numpyro.distributions import constraints
from tqdm import tqdm
import argparse


class TBIP:
    """Define the model and variational family"""
    def __init__(
        self, 
        N, 
        D, 
        K, 
        V, 
        batch_size, 
        init_mu_theta=None, 
        init_mu_beta=None
    ):
        self.N = N # number of people
        self.D = D # number of documents
        self.K = K # number of topics
        self.V = V # number of words in vocabulary
        self.batch_size = batch_size # number of documents in a batch

        if init_mu_theta is None:
            init_mu_theta = jnp.zeros([D, K])
        else:
            self.init_mu_theta = init_mu_theta

        if init_mu_beta is None:
            init_mu_beta = jnp.zeros([K, V])
        else:
            self.init_mu_beta = init_mu_beta

    def model(self, Y_batch, d_batch, i_batch):

        with plate("i1", self.N, dim = -2):
            with plate("i2", 2, dim = -1):
                x = sample("x", dist.Normal())
        # print('x_shape', x.shape)

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                # beta = sample("beta", dist.Gamma(0.3, 0.3))
                beta = sample("beta", dist.Gamma(2, 1))
        # print('beta_shape', beta.shape)

        with plate("k", size = self.K, dim = -3):
            with plate("k_v", size = self.V, dim = -2):
                with plate("k_v_2", 2, dim = -1):
                    eta = sample("eta", dist.Normal())

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                # Sample document-level latent variables (topic intensities)
                # theta = sample("theta", dist.Gamma(0.3, 0.3))
                theta = sample("theta", dist.Gamma(2, 1))

            # Compute Poisson rates for each word
            # P = jnp.sum(jnp.expand_dims(theta, 2) * jnp.expand_dims(beta, 0) *
            #     jnp.exp(jnp.expand_dims(x[i_batch], (1,2)) * jnp.expand_dims(eta, 0)), 1)
            
            # print('x[i_batch]', x[i_batch].shape,'eta', eta.shape)
            # print(jnp.expand_dims(x[i_batch], (1,2)).shape)

            P = jnp.sum(jnp.expand_dims(theta, axis=2) * jnp.expand_dims(beta, axis=0) *
                jnp.exp(jnp.dot(x[i_batch], eta.transpose(0, 2, 1))), axis=1)

            with plate("v", size = self.V, dim = -1):
                # Sample observed words
                sample("Y_batch", dist.Poisson(P), obs = Y_batch)

    def guide(self, Y_batch, d_batch, i_batch):
        # This defines variational family. Notice that each of the latent variables
        # defined in the sample statements in the model above has a corresponding
        # sample statement in the guide. The guide is responsible for providing
        # variational parameters for each of these latent variables.

        # Also notice it is required that model and the guide have the same call
        mu_x = param("mu_x", init_value = -1  + 2 * random.uniform(random.PRNGKey(1), (self.N, 2)))
        sigma_x = param("sigma_y", init_value = jnp.ones([self.N, 2]), constraint  = constraints.positive)

        mu_eta = param("mu_eta", init_value = random.normal(random.PRNGKey(2), (self.K, self.V, 2)))
        sigma_eta = param("sigma_eta", init_value = jnp.ones([self.K,self.V, 2]), constraint  = constraints.positive)

        mu_theta = param("mu_theta", init_value =  self.init_mu_theta)
        sigma_theta = param("sigma_theta", init_value =  jnp.ones([self.D, self.K]), constraint  = constraints.positive)

        mu_beta = param("mu_beta", init_value = self.init_mu_beta)
        sigma_beta = param("sigma_beta", init_value = jnp.ones([self.K, self.V]), constraint  = constraints.positive)

        with plate("i1", self.N, dim = -2):
            with plate("i2", 2, dim = -1):
                sample("x", dist.Normal(mu_x, sigma_x))

        with plate("k", size = self.K, dim = -2):
            with plate("k_v", size = self.V, dim = -1):
                sample("beta", dist.LogNormal(mu_beta, sigma_beta))
                
        with plate("k", size = self.K, dim = -3):
            with plate("k_v", size = self.V, dim = -2):
                with plate("k_v_2", 2, dim = -1):      
                    sample("eta", dist.Normal(mu_eta, sigma_eta))

        with plate("d", size = self.D, subsample_size=self.batch_size, dim = -2):
            with plate("d_k", size = self.K, dim = -1):
                sample("theta", dist.LogNormal(mu_theta[d_batch], sigma_theta[d_batch]))

    def get_batch(self, rng, Y, author_indices):
        # Helper functions to obtain a batch of data, convert from scipy.sparse to jax.numpy.array and move to gpu
        D_batch = jax.random.choice(rng, jnp.arange(self.D), shape = (self.batch_size,))
        Y_batch = jax.device_put(jnp.array(Y[D_batch].toarray()), jax.devices("gpu")[0])
        D_batch = jax.device_put(D_batch, jax.devices("gpu")[0])
        I_batch = jax.device_put(author_indices[D_batch], jax.devices("gpu")[0])
        return Y_batch, I_batch, D_batch


def main(args):

    # Constants
    orig = args.orig
    save = args.save

    # Load data
    author_indices = jax.device_put(
        jnp.load(orig + "author_indices.npy"), jax.devices("gpu")[0]
    )
    counts = sparse.load_npz(orig + "counts.npz")

    with open(orig + "vocabulary.txt",'r') as f:
        vocabulary=f.readlines()

    with open(orig + "author_map.txt",'r') as f:
        author_map=f.readlines()


    # Params
    author_map = np.array(author_map)
    num_authors = int(author_indices.max() + 1)
    num_documents, num_words = counts.shape
    pre_initialize_parameters = args.pre_initialize_parameters
    num_topics = args.num_topics
    rng_seed = random.PRNGKey(args.rng_seed)
    seed = args.seed
    iter = args.iter
    num_steps = args.num_steps
    batch_size = args.batch_size
    lr = args.lr
    decay = args.decay
    print_steps = args.print_steps
    print_termi = args.print_termi
    words_per_topic = args.words_per_topic

    if pre_initialize_parameters:
        nmf_model = NMF(
            n_components=num_topics,
            init='random',
            random_state=seed,
            max_iter=iter
        )
        initial_document_loc = jnp.log(
            jnp.array(np.float32(nmf_model.fit_transform(counts) + 1e-2))
        )
        initial_objective_topic_loc = jnp.log(
            jnp.array(np.float32(nmf_model.components_ + 1e-2))
        )
    else:
        rng1, rng2 = random.split(rng_seed, 2)
        initial_document_loc = random.normal(
            rng1, shape = (num_documents, num_topics)
        )
        initial_objective_topic_loc = random.normal(
            rng2, shape =(num_topics, num_words)
        )

    tbip = TBIP(
        N=num_authors,
        D=num_documents,
        K=num_topics,
        V=num_words,
        batch_size=batch_size,
        init_mu_theta=initial_document_loc,
        init_mu_beta=initial_objective_topic_loc
    )

    svi_batch = SVI(
        model=tbip.model,
        guide=tbip.guide,
        optim = adam(exponential_decay(lr, num_steps, decay)),
        loss = TraceMeanField_ELBO()
    )

    # Compile update function for faster training
    svi_batch_update = jit(svi_batch.update)

    # Get initial batch. This informs the dimension of arrays and ensures they are
    # consistent with dimensions (N, D, K, V) defined above.
    Y_batch, I_batch, D_batch = tbip.get_batch(
        random.PRNGKey(1), counts, author_indices
    )

    # Initialize the parameters using initial batch
    svi_state = svi_batch.init(
        random.PRNGKey(0),
        Y_batch = Y_batch,
        d_batch = D_batch,
        i_batch = I_batch
    )

    def get_topics(neutral_mean,
                    negative_mean,
                    positive_mean,
                    vocabulary,
                    print_to_terminal=print_termi,
                    words_per_topic=words_per_topic):
        num_topics, num_words = neutral_mean.shape
        words_per_topic = words_per_topic
        top_neutral_words = np.argsort(-neutral_mean, axis=1)
        top_negative_words = np.argsort(-negative_mean, axis=1)
        top_positive_words = np.argsort(-positive_mean, axis=1)
        topic_strings = []
        for topic_idx in range(num_topics):
            neutral_start_string = "Neutral  {}:".format(topic_idx)
            neutral_row = [vocabulary[word] for word in
                            top_neutral_words[topic_idx, :words_per_topic]]
            neutral_row_string = ", ".join(neutral_row)
            neutral_string = " ".join([neutral_start_string, neutral_row_string])

            positive_start_string = "Positive {}:".format(topic_idx)
            positive_row = [vocabulary[word] for word in
                            top_positive_words[topic_idx, :words_per_topic]]
            positive_row_string = ", ".join(positive_row)
            positive_string = " ".join([positive_start_string, positive_row_string])

            negative_start_string = "Negative {}:".format(topic_idx)
            negative_row = [vocabulary[word] for word in
                            top_negative_words[topic_idx, :words_per_topic]]
            negative_row_string = ", ".join(negative_row)
            negative_string = " ".join([negative_start_string, negative_row_string])

            if print_to_terminal:
                topic_strings.append(negative_string)
                topic_strings.append(neutral_string)
                topic_strings.append(positive_string)
                topic_strings.append("==========")
            else:
                topic_strings.append("  \n".join(
                    [negative_string, neutral_string, positive_string]))

        if print_to_terminal:
            all_topics = "{}\n".format(np.array(topic_strings))
        else:
            all_topics = np.array(topic_strings)
        return all_topics


    # Run SVI
    rngs = random.split(random.PRNGKey(2), num_steps)
    losses = []
    pbar = tqdm(range(num_steps))
    for step in pbar:
        Y_batch, I_batch, D_batch = tbip.get_batch(rngs[step], counts, author_indices)
        svi_state, loss = svi_batch_update(svi_state,
            Y_batch = Y_batch,
            d_batch = D_batch,
            i_batch = I_batch)

        loss = loss/counts.shape[0]
        losses.append(loss)
        if step%print_steps == 0 or step == num_steps - 1:
            pbar.set_description("Init loss: " + "{:10.4f}".format(jnp.array(losses[0])) +
            "; Avg loss (last 100 iter): " + "{:10.4f}".format(jnp.array(losses[-100:]).mean()))

        if (step + 1) % 2500 == 0 or step == num_steps - 1:

            print(f"Results after {step} steps.")
            estimated_params = svi_batch.get_params(svi_state)
            
            neutral_mean = estimated_params["mu_beta"] + estimated_params["sigma_beta"]**2/2

            positive_mean = (estimated_params["mu_beta"] + estimated_params["mu_eta"].mean(axis = -1) +
                (estimated_params["sigma_beta"]**2 + estimated_params["sigma_eta"].mean(axis = -1)**2 )/2)

            negative_mean = (estimated_params["mu_beta"] - estimated_params["mu_eta"].mean(axis = -1) +
                (estimated_params["sigma_beta"]**2 + estimated_params["sigma_eta"].mean(axis = -1)**2 )/2)
            
            os.makedirs(save, exist_ok=True)
            np.save(f"{save}neutral_topic_mean.npy", neutral_mean)
            np.save(f"{save}negative_topic_mean.npy", positive_mean)
            np.save(f"{save}positive_topic_mean.npy", negative_mean)

            topics = get_topics(
                neutral_mean,
                positive_mean,
                negative_mean,
                vocabulary
            )

            with open(f"{save}topics.txt", 'w') as f:
                print(topics, file=f)
            print(topics)

            authors = pd.DataFrame({"name": author_map, "ideal_point_x1": estimated_params["mu_x"][:,0], "ideal_point_x2": estimated_params["mu_x"][:,1]})
            authors.to_csv(f"{save}authors.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TBIP model')
    parser.add_argument('--orig', type=str, default='data/clean_full_2/', help='Path to the original data directory.')
    parser.add_argument('--save', type=str, default='data/save/save_0/', help='Path where the outputs will be saved.')
    parser.add_argument('--pre_initialize_parameters', type=bool, default=True, help='Whether to pre-initialize parameters using NMF.')
    parser.add_argument('--num_topics', type=int, default=50, help='Number of topics.')
    parser.add_argument('--rng_seed', type=int, default=0, help='Random seed for PRNG keys.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for random state in NMF.')
    parser.add_argument('--iter', type=int, default=500, help='Number of iterations for NMF.')
    parser.add_argument('--num_steps', type=int, default=50000, help='Number of steps for SVI.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for documents.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimizer.')
    parser.add_argument('--decay', type=float, default=0.01, help='Decay rate for learning rate decay.')
    parser.add_argument('--print_steps', type=int, default=10000, help='Frequency of logging during SVI.')
    parser.add_argument('--print_termi', type=bool, default=True, help='Whether to print topics to terminal.')
    parser.add_argument('--words_per_topic', type=int, default=10, help='Number of words to display per topic.')
    args = parser.parse_args()

    main(args)