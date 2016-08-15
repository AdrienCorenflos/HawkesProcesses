# HawkesProcesses
Package to simulate and fit Hawkes processes for time arrival phenomenas


## I. Definitions

### 1. Definition of a self-exciting process

A self-exciting process can be defined as a jump process whose jump intensity (probability of instantaneous jump depends on the jumps it has previously had).

The definition of a general self-exciting process is the following:
$$\lambda(t) = \lambda_0(t) + \int_{0}^t\nu(t − s)dN_s=\lambda_0(t) + \sum_{t_i \in J_t}\nu(t − t_i ),$$
whith $\lambda_0 : \mathbb{R}\to\mathbb{R}_+$ deterministic initial intensity for the process $J_t$ the set of jump times $t_i < t$ and $\nu : \mathbb{R}_+\to\mathbb{R}_+$
a positive self-excitement function of the past events $t_i$, and $s \mapsto N_s$ is a standard poisson process.
Namely, the closer you are from previous jumps, the more likely you are to jump again.

### 2. Definition of a linear self-exciting process

The linearity comes from the fact that jumps are supposed to depend linearily on previous jumps.
A linear self-exciting process (Hawkes) is a process whose self-excitement function $\nu$ reads:
$$ \nu(t) = \alpha e^{−\beta t}\mathbb{1}_{\mathbb{R}_+}$$

And whose initial intensity is constant:
$$\lambda_0(t) = \lambda_0$$

This implies the following form for the self exciting process:

$$\lambda(t) = \lambda_0 + \int_0^t\alpha e^{−\beta(t−s)}dN_s=\lambda_0+\sum_{t_i<t}\alpha e^{-\beta(t-t_i)}.$$

### 3. Intuitions

A poisson process is a discrete process living in the space $\mathbb{N}$. Its intensity defines the probability of an instantaneous jump. Namely, if $N$ is a poisson process of say intensity $3$, then by definition:

$$ \mathbb{P}(N_{t+h} - N{t} = 1) = 3 \times h +o(h)$$

and the probability of having more than one jump is neglectible when $h << 1$:

$$ \mathbb{P}(N_{t+h} - N{t} > 1) = o(h)$$

The intuition is the same for a path dependant intensity such as ours.

### 4. Usage

You can use Hawkes processes for anything where you believe there could be a self-excitement process undergoing:
    - Tweets arrivals
    - Market orders
    - Crowd panic
    - Etc
    
## II. Implementation

### 1. Prerequisites

A theorem states that for the process to be asymptotically stationary, one needs to impose $\alpha < \beta$.
This is to say that after a long period without jumps, the process returns to its initial intensity.

### 2. Algorithm

See [Ogata](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1056305&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel5%2F18%2F22718%2F01056305.pdf%3Farnumber%3D1056305) for details

In plain words the algorithm is as follow:
Start from a given intensity and position, as the interval jump times are distributed as exponential random variables, simulate one with the intensity you've been given.
Now your new jump point is the one you've just been given. 
Then comes the tricky part, we need to update the intensity.
Set up the new intensity at the very moment of the jump, now simulate the next jump as if this new intensity was to remain constant.
Two cases are now available:
    - The new jump could have happened in the path dependant framework, then accept it and continue.
    - The new jump couldn't have been accepted in the path dependant framework, then reject it and continue.
    
    
#### a. Initialisation
$\lambda = \lambda_0$
#### b. First jump
Generate $e = Exp(\lambda)$ and set $s\leftarrow e$, this is your exponential.

If $s\leq T$,

   Then $t_1\leftarrow s$,
    
Else you're done.
#### c. General algorithm
_a. Update the intensity_: Set $\lambda\leftarrow\lambda(t_{n−1}) + \alpha$. 
$\lambda$ shows a positive excitment of $\alpha$ as a jump just happened


_b. New event_ : $e = Exp(\lambda)$ and set $s\leftarrow s+e$.

**If** $s\geq T$,

**Then** you're done.

_c. Rejection test_ : Generate $D\sim U_{[0,1]}$.

**If** $D\leq\frac{\lambda(s)}{\lambda}$,

**Then** $t_n\leftarrow s$ and go through the general routine again,

**Else** update $\lambda\leftarrow\lambda(s)$ and try a new date at step b. of the general
routine.    


## III. Multidimensional Hawkes processes

### 1. Definition

The previous case of a linear self-exciting Hawkes process presupposes the same influence for all jumps. It moreover can't take into account influence by jumps from other sources : TF1 jump could impact Bouygue.
This is what we'll assess in the following definition

We will denote $N_t = (N^1_t,\ldots,N^M_t)$ the multidimensional counting process considered.

P denotes the number of influences a past jump can have.

A multidimensional Hawkes process with multiple jump influences is defined by intensities $\lambda^m$, $m = 1,\ldots,M$ given by
$$\lambda^m(t) = \lambda^m_0 + \sum_{n=1}^M\int_0^t\sum_{j=1}^P\alpha^{mn}_j e^{−\beta^{mn}_j(t−s)}dN^n_s,$$
$i.e.$ in its simplest version with $P = 1$
$$\lambda^m(t) = \lambda^m_0 + \sum_{n=1}^M\int_0^t\alpha^{mn} e^{−\beta^{mn}(t−s)}dN^n_s = \lambda^m_0 + \sum_{n=1}^M\sum_{t^n_i<t}\alpha^{mn} e^{−\beta^{mn}(t−t_i^n)}.$$

### 2. Algorithm

Let $t^m$ be the last time of jump for the mth component.

Let $\lambda^M(t) = \sum_{m=0}^M \lambda^m(t)$.

#### a. Initialisation
$\lambda \leftarrow \lambda^M(0)$
#### b. First jump
Generate $e = Exp(\lambda)$ and set $s\leftarrow e$, this is your exponential time of jump arrival.

If $s \gt T$, you're done.

Else:

Acceptance test:

_a. Generate $D\sim U_{[0,1]}$

_b. Set $n_0$ such that

$ \lambda^{n_0-1}(0) \lt D  . \lambda \leq \lambda^{n_0}(0) $.

The jump is for component $n_0$.

_c. $t^{n_0} \leftarrow t_1 \leftarrow s$

#### c. General algorithm
_a. Update the intensity_: Set $\lambda \leftarrow \lambda^M(t_{i−1}) + \sum_{m=1}^M \alpha^{n n_0}$. 


_b. New event_ : $e = Exp(\lambda)$ and set $s\leftarrow s+ e$.

**If** $s\geq T$,

**Then** you're done.

_c. Rejection test_ : Generate $D\sim U_{[0,1]}$.

**If** $D\leq\frac{\lambda(s)}{\lambda}$,

**Then** 
- Set $n_0$ such that $ \lambda^{n_0-1}(s) \lt D  . \lambda \leq \lambda^{n_0}(s) $.

- $t^{n_0} \leftarrow t_i \leftarrow s$

**Else** update $\lambda \leftarrow \lambda^M(s)$ and try a new date at step b. of the general
routine.


## III. Price model

We can define a price model as :
$$p(t) = N^1(t) - N^2(t)$$
where
$N^1$ and $N^2$ are Hawkes processes with intensities given by:
    $$ \lambda^1(t) = \lambda_0 + \int_{0}^t \alpha \exp(-\beta(t-s)) dN^2_s \\
    \lambda^2(t) = \lambda_0 + \int_{0}^t \alpha \exp(-\beta(t-s)) dN^1_s
    $$
    
This is a mean reversion model for microstructure


    
    
