# Waypoint planning algorithms

As randomly sampling a waypoint from the workplace of a robotic arm
would produce "silly" and unnecessarily lengthy detours while offering
no control over representation of colliding vs. non-colliding trajectories
in generated dataset, an approach using analytical geometry was chosen
to minimize the unnatural detour planning as well as providing an option
to steer away from the overabundant presence of any one type of trajectory in the dataset.

0. We define:
   - start position of the end effector $\mathbf{S} \in \mathbb{R}^3,$
   - goal position of the end effector $\mathbf{G} \in \mathbb{R}^3,$
   - Distractor centre approximated by a sphere $\mathbf{D} \in \mathbb{R}^3,$
   - Sphere's radius $R > 0,$
   - Base clearance margin to stay away from the occlusion $c_{base} > 0,$
   - Safety margin $m_{safe} > 0.$

### TLDR

Detour waypoint is given as:
$$ \mathbf{W} = \mathbf{S} + t_{anchor}(\mathbf{G} - \mathbf{S}) + (R + c_{base})s_r \cdot (\cos{\theta \hat{\mathbf{u}}} + \sin{\theta \hat{\mathbf{w}}}),$$
$$ ||\mathbf{W} - \mathbf{D}|| \ge R + m_{safe},
||\mathbf{W} - \mathbf{S}|| + ||\mathbf{G} - \mathbf{W}|| \le \alpha ||\mathbf{G} - \mathbf{S}||, \alpha \in \mathbb{R}^+$$
where:
$$ t_{anchor} \sim \mathcal{N}\left(clip\left(\frac{(\mathbf{D} - \mathbf{S})\cdot(\mathbf{G}-\mathbf{S})}{||\mathbf{G} - \mathbf{S}||^2}, 0, 1\right), \sigma^{2}_{t}\right), t_{anchor} \in [t_{min}, t_{max}] \subset (0, 1) $$
$$ s_r \sim \mathcal{Uniform}(s_{min}, s_{max}), [s_{min}, s_{max}] \subseteq \mathbb{R}^+ $$
$$ \theta \sim \mathcal{Uniform}(0, 2\pi) $$
$$ \hat{\mathbf{u}}, \hat{\mathbf{w}} \text{ describe a plane orthogonal to $\mathbf{S}$->$\mathbf{G}.$} $$

Impact point is given as:
$$ \mathbf{P}_{impact} = \mathbf{D} + r\hat{\mathbf{v}}, $$
where:
$$ r \sim \mathcal{Uniform}(0.2R, 0.8R) $$
$$ \hat{\mathbf{v}} \in Cone = \left\{\hat{\mathbf{v}}\mid ||\hat{\mathbf{v}}|| = 1
\land \hat{\mathbf{v}} \cdot 
\frac{\mathbf{D} - \mathbf{S}}{||\mathbf{D}-\mathbf{S}||} \ge 
\cos{\phi_{max}}\right\}, \phi_{max} \in (0, 2\pi)$$

## Detour (Avoidance) Waypoint

1. We calculate direction (tangent) $\hat{\mathbf{t}} \in \mathbb{R}^3$ of the direct path $\mathbf{S}$->$\mathbf{G}.$
$$ \hat{\mathbf{t}} = \frac{\mathbf{v}}{||\mathbf{v}||}, \text{where } \mathbf{v} = \mathbf{G} - \mathbf{S}$$

2. We project the centre $\mathbf{D}$ onto line $\mathbf{S}$->$\mathbf{G}$ using anchor parameter $t_{closest}$ to find the closest point of the line
to the distractor $\mathbf{C}.$
$$ t_{closest} = \frac{(\mathbf{D} - \mathbf{S})\cdot(\mathbf{G}-\mathbf{S})}{||\mathbf{G} - \mathbf{S}||^2} $$
$$ t'_{closest} = clip(t_{closest}, 0, 1) $$
$$ \mathbf{C} = \mathbf{S} + t'_{closest}(\mathbf{G} - \mathbf{S}) $$

3. We introduce randomness by randomly choosing a point from interval near the closest point
by moving the anchor parameter.
$$ t_{anchor} \sim \mathcal{N}(t'_{closest}, \sigma^{2}_{t}), t_{anchor} \in [t_{min}, t_{max}] \subset (0, 1) $$
$$ \mathbf{C} = \mathbf{S} + t_{anchor}(\mathbf{G} - \mathbf{S}) $$

4. We define vectors $\hat{\mathbf{u}}, \hat{\mathbf{w}}$ that span a plane perpendicular to the path.
Let $\mathbf{a}$ be a vector that is not colinear with $\hat{\mathbf{t}},$ then:
$$ \mathbf{u} = \hat{\mathbf{t}} \times \mathbf{a}, \hat{\mathbf{u}} = \frac{\mathbf{u}}{||\mathbf{u}||}$$
$$ \mathbf{w} = \hat{\mathbf{t}} \times \hat{\mathbf{u}} , \hat{\mathbf{w}} = \frac{\mathbf{w}}{||\mathbf{w}||}$$

5. We push the waypoint away from the distractor by distance $r$ which is given by clearance $c$ randomized
by a given magnitude $s_r.$
$$ c = R + c_{base} $$
$$ r = c \cdot s_r, s_r \sim \mathcal{Uniform}(s_{min}, s_{max}), [s_{min}, s_{max}] \subseteq \mathbb{R}^+ $$

6. We also randomize the direction of the push by sampling an angle from the full circle and define direction
vector $\hat{\mathbf{d}}(\theta)$ that is orthogonal to $\hat{\mathbf{t}}.$
$$ \theta \sim \mathcal{Uniform}(0, 2\pi) $$
$$ \hat{\mathbf{d}}(\theta) = \cos{\theta \hat{\mathbf{u}}} + \sin{\theta \hat{\mathbf{w}}}$$

7. Lastly, we check if the push is valid, i.e. waypoint $\mathbf{W}$ is valid,
and truly leads to potentially avoidance while still using physics engine as ground truth.
$$ \mathbf{W} = \mathbf{C} + r\hat{\mathbf{d}}(\theta) $$
$$ \text{If } ||\mathbf{W} - \mathbf{D}|| < R + m_{safe} \text{, then $\mathbf{W}$ is not valid,
 resample $\theta, r$}$$
$$ \text{Path-length check: } L_{direct} = ||\mathbf{G}-\mathbf{S}||, L_{detour} = ||\mathbf{W} - \mathbf{S}|| + ||\mathbf{G} - \mathbf{W}||$$
$$\text{Accept only if } L_{detour} \le \alpha L_{direct}, \alpha \in \mathbb{R}^+$$

## Impact (Collision) Waypoint

1. We define direction from the starting position to the distractor $\hat{\mathbf{d}}.$
$$ \hat{\mathbf{d}} = \frac{\mathbf{D} - \mathbf{S}}{||\mathbf{D}-\mathbf{S}||}$$

2. We find a new direction $\hat{\mathbf{v}}$ that is within an angle $\phi_{max}$ of $\hat{\mathbf{d}}$ by sampling
unit vectors until following condition is met.
$$ \hat{\mathbf{v}} \cdot \hat{\mathbf{d}} \ge \cos{\phi_{max}}, \hat{\mathbf{v}} \text{ is a unit vector} $$

3. We find an impact point $P_{impact}$ that is randomly near or on the surface of the distractor, given by approximate
radius $r$, which potentially leads to a colliding trajectory.
$$ r \sim \mathcal{Uniform}(0.0R, 0.2R) $$
$$ \mathbf{P}_{impact} = \mathbf{D} + r\hat{\mathbf{v}} $$
