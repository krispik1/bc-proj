### RRT planning algorithm

Rapidly-exploring Random Tree (RRT) algorithms provide more structural approach
to generating trajectories, where nodes represent joint configurations and each 
step is represented by an edge between joint configuration the robot starts with 
in a given state to joint configuration the robot would reach in the following state.
By randomly sampling joint configuration space, allowing only sensible movements,
a new trajectory is created step by step and is steered towards the goal configuration
which is given by the desired end effector position (i.e. either goal object position or
an impact point with the occlusion).

We define:
- Number of the DoF of the utilized robotic arm $n,$
- Upper limit for each joint $highs = \left( \theta_{max_1}, \dots, \theta_{max_n} \right),$
- Lower limit for each joint $lows = \left( \theta_{min_1}, \dots, \theta_{min_n} \right),$
- Configuration space $\mathcal{C} = \left\{ (\theta_1, \dots, \theta_n) \mid \theta_i \in \left[ \theta_{min_i}, \theta_{max_i} \right], i \in [1, \dots, n] \right\},$
- Start and goal joint angle configuration $q_{start}, q_{goal} \in \mathcal{C},$ 
- Step size $\epsilon \in \mathbb{R},$
- Goal bias (probability of sampling the goal configuration) $p \in [0, 1],$
- Number of substeps in one edge $k,$
- Tree data structure represented as list of nodes and indices of parents $\mathcal{T}.$

1. We need to sample a configuration $q_{sample}:$
$$ q_{sample} = 
\begin{cases}
q & \text{with probability $1-p$ sampled from $C$}\\
q_{goal} & \text{with probability $p$}
\end{cases}
$$

2. We find the nearest configuration $q_{near} \in \mathcal{T}$ to $q_{sample}:$
$$ q_{near} = argmin_{q \in \mathcal{T}} dist(q, q_{sample}) $$

3. We steer edge towards the sampled configuration $q_{sample}$ from where we get new
configuration $q_{new}$
$$ q_{new} = q_{near} + \frac{q_{sample} - q_{near}}{dist(q_{near}, q_{sample}) \cdot min(\epsilon, dist(q_{near}, q_{sample}))}$$

4. We check validity of the edge by checking validity for every of the $k$ substeps. This depends on boolean $allow\_hit.$
We simulated each substep configuration $q$ and check if it leads to collision getting collision flag $collision\_flag(q).$
Finally, we check if the edge is valid getting $valid(q_{near}, q_{new}).$
$$
allow\_hit = 
\begin{cases}
1 & \text{Used in collide mode of planner, can collide}\\
0 & \text{otherwise}
\end{cases}
$$
$$
collision\_flag(q) = 
\begin{cases}
1 & \text{Configuration $q$ leads to collision}\\
0 & \text{otherwise}
\end{cases}
$$
$$
valid(q_{near}, q_{new})=argmin_{q \text{ is substep of edge (q_{near}, q_{new})}}(\neg{collision\_flag(q) \lor allow\_hit})
$$

5. If the edge is valid, $q_{new}$ is added as a new node to the tree $\mathcal{T}.$

6. If planner is in collide mode and there was collision, we traverse $\mathcal{T}$ from
$q_{new}$ through parents until we get to $q_{start}$. This traversal in reverse is our colliding trajectory.

7. Otherwise, if $q_{new}$ is near enough $q_{goal}$ and $valid(q_{new}, q_{goal}) == 1,$ $q_{new}$ becomes parent of
$q_{goal}$, and we again traverse the tree similarly to last step starting from $q_{goal}.$

8. If neither condition is satisfied, there was no trajectory found in our set time/node limit.





