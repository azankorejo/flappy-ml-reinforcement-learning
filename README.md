```
This project is all about building a game inspired by the Flappy Bird classic, using Python
and the pygame library. The focus here was on refactoring the code to follow an object-
oriented approach, which helps make the game more manageable and scalable. By
breaking the code down into classes and methods, we improved readability and made it
easier to adjust different game mechanics.


```
```
The game itself is pretty simple—control a bird trying to fly through pairs of pipes without
hitting them or the ground. You can "flap" the bird upwards, and you need to avoid
obstacles as they move across the screen. The longer you survive, the higher your score.
```
```
2.1. Main Game Loop

The game runs on a loop, which is controlled by the FPS (frames per second). The frame
rate here is set to 30 FPS, which keeps the gameplay smooth and consistent. The main
loop does everything
```
```
1 It listens for user input (flapping the bird)^
1 It checks for collisions with pipes or the ground^
1 It updates the bird’s position and pipes' movement^
1 It refreshes the screen with new graphics and updates the score.


```
```
2.2. Object-Oriented Design

Instead of having one giant block of code, the project is split up into different classes. The
central class is the GameState, which controls the game’s state, like whether the game is
running, paused, or over. Each object (bird, pipes, score, etc.) is now its own class, making
the game easier to manage and extend.

For example
1 The bird is represented by a class that manages its movement, collisions, and sprite
(image) handling^
1 Pipes are another class that handles generating pipes, moving them, and checking for
collisions^
1 The game state keeps track of the score, handles pausing, and resets the game when
the bird dies.
```
## 1. Project Overview

## 2. How the Game Works

# Deep Reinforcement Learning On Flappy bird

# Game Project Report^1 Azan Korejo

```
2.3. Asset Management (Sprites and Images)

The game needs images for everything—backgrounds, the bird, the pipes, etc. To keep
things organized and efficient, I loaded all these assets into a dictionary in the load_assets
function. Each asset (like the bird or pipe image) gets stored in a dictionary key, so we can
easily access and replace them if needed.


```
```
The key to making it all work smoothly was hitmasking. This is a fancy way of saying that
each image has an invisible mask over it that helps the game check for collisions in a
more accurate way than just using bounding boxes.


```
```
2.4. Collision Detection

Now, let’s talk about how the game knows when the bird hits a pipe or the ground. Initially,
we use rectangular bounding boxes around the bird and pipes, but that’s not super
accurate. So, for pixel-perfect accuracy, I used hitmasking.
```
```
This method checks the actual pixels in the bird and pipe images to see if they overlap. It’s
more precise and works better than just relying on rectangle sizes.


```
```
2.5. Random Pipe Generation

One of the most important parts of the game is the pipes. They need to move across the
screen, but also appear randomly, so the game doesn’t feel repetitive. The gap between
pipes is randomized, and the pipes themselves scroll from right to left, creating a sense of
continuous motion.


```
```
The game generates new pipes when the old ones go off-screen, keeping the flow going.
You never know where the pipes are going to show up, which keeps things interesting.


```
```
2.6. Scoring System

The score increases every time the bird successfully passes through a set of pipes. The
score is calculated based on the bird’s x-position in relation to the pipes. If the bird moves
past the middle of a pipe, the score goes up.


```
```
2.7. Bird Physics

The bird’s movement is controlled by two forces: gravity and the flap. Gravity pulls the bird
down over time, while the flap makes it rise. The bird’s velocity is adjusted each frame
based on these forces. If the bird flaps, it gets an upward speed boost, and if it doesn’t,
gravity pulls it down.

The movement looks natural because of the way the physics are modeled—it’s not just a
simple jump; it feels like the bird is actually flying.
```
```
3.1. Accurate Collision Detection

The hardest part of this game was getting the pixel-perfect collision detection right. It's
tricky to compare pixels from two images, and it requires a bit more processing power
than regular rectangle collision detection. But it’s worth it because it makes the game feel
much more polished and real.


```
```
3.2. Random Pipe Generation and Balancing Difficulty

Making the game challenging yet fair is a tough balance. The pipes have to be random, but
if the gap is too wide or too narrow, it can make the game either too easy or frustratingly
hard. Finding that sweet spot where the game feels challenging but not impossible took
some trial and error.


```
```
3.3. Smooth Game Loop and FPS Control

Another challenge was making sure the game ran smoothly at 30 FPS. If there’s any lag,
the game feels jerky, and that messes with player experience. Keeping everything running
at a steady FPS, even when loading assets and performing calculations, took some
optimization.

```
```
4.1. Caching Assets

Since loading images and sounds can take time, I made sure to cache the assets in
memory. This way, they are only loaded once at the start of the game, instead of every
time they are used. It keeps things fast and reduces unnecessary disk I/O.


```
```
4.2. Modularizing the Code

By using object-oriented principles, the game’s code is more organized. I broke the game
down into smaller, manageable pieces like the bird, the pipes, and the game state. This
makes it way easier to add new features, fix bugs, or tweak gameplay without messing
with the whole system.


```
```
4.3. Frame-Rate Consistency

To keep the game feeling smooth, I used a frame-rate cap of 30 FPS. This ensured the
game runs consistently on most systems, preventing things from speeding up or slowing
down randomly.
```
## 3. Challenges Faced

## 4. Optimization Techniques

```
Deep Q-Network (DQN) Overview

In our Flappy Bird project, we use a reinforcement learning algorithm called Deep Q-
Network (DQN), which enables the bird to learn how to navigate the pipes through trial and
error. Here’s a breakdown of how DQN is applied here
? Q-Values and Q-Learning: Q-values represent the expected utility (or reward) for each
action the bird can take at each possible state of the game. The goal of Q-learning is to
learn the Q-values by balancing exploration (trying new actions) and exploitation
(choosing the best-known actions)^
6 Deep Q-Network (DQN): In DQN, instead of storing Q-values for every state-action pair
(which is impossible for complex environments), we use a neural network to
approximate Q-values. Here, the network maps the current game state to Q-values for
each action (flap or no-flap)^
m Experience Replay: During training, each frame (state) and the corresponding actions
and rewards are stored in memory, allowing the agent to learn from past experiences.
The DQN replays these stored experiences in batches, which helps in reducing data
correlation and stabilizing learning^
 Target Network: To stabilize training, a separate target network is used, which is
updated less frequently than the main network. This target network helps in more
consistent Q-value updates by reducing oscillations in the Q-learning process.


```
```
Algorithm Implementation Details

The DQN algorithm is implemented using the following approach
? Initialize Networks: Two neural networks are set up—a primary network that
approximates Q-values and a target network for stabilizing learning. Both networks
have weights initialized randomly^
6 Define Rewards and States: Rewards in the Flappy Bird environment are
O + 1 reward when the bird passes a pipe (success)^
O +0.1 reward for survival (to encourage longer survival)^
O -1 reward when the bird crashes^
m Training Steps(
1 Each frame represents a single time step where the bird observes its current state^
1 The agent chooses an action based on the current Q-values^
1 If the agent flaps, it receives an updated vertical velocity and position^
1 After the action is taken, the agent observes the new state and receives a reward^
1 The experience is stored in memory^
1 The network is trained by randomly sampling experiences from memory to avoid
correlation^
1 The Q-values are updated by minimizing the difference between predicted Q-values
and target Q-values calculated from the target network.
```
## 5. Reinforcement Learning Approach

```
 Exploration and Exploitation: We use an epsilon-greedy approach where the bird
randomly chooses actions with a certain probability. Over time, epsilon (exploration
rate) decreases, so the agent relies more on learned Q-values rather than random
actions.
```
## Challenges Faced in Deep Reinforcement Learning (DRL)

```
This project’s DRL component, specifically in using Deep Q-Networks (DQN) for
Flappy Bird, introduced a few key challenges that required special attention
? Exploration-Exploitation Balance:
The decision-making process in reinforcement learning hinges on balancing
exploration (trying new moves) with exploitation (repeating known successful
actions). In a game like Flappy Bird, where the objective is to avoid pipes while
maximizing score, over-exploring can lead to numerous failures without much
reward, while over-exploiting can make the agent fail to generalize effectively to
various pipe configurations. Adjusting the epsilon-greedy policy parameters to
achieve this balance was a meticulous task^
6 Sparse Rewards:
Rewards in the game are sparse—only granted when passing through pipes—
making it difficult for the model to learn effectively in early stages. This required
reward shaping, where incremental rewards were given based on proximity to
pipes, which encouraged the agent to survive longer. This technique helped
provide the model with enough feedback to understand that staying alive longer
was part of achieving higher rewards^
m Non-Stationary Environment:
The environment in Flappy Bird changes dynamically with each frame, as pipes
move towards the player, who also moves up or down. This non-stationarity can
make it hard for the DQN agent to learn stable policies, as it constantly needs to
adapt to changes in state values over time. We addressed this by using
experience replay and updating the Q-network less frequently to smooth out the
learning process^
 Training Instability:
DQNs are known to be unstable during training, especially with complex
environments. The fluctuating nature of Flappy Bird’s gameplay (e.g., the random
placement of pipes) increased the volatility in learning, often leading to erratic
performance across episodes. To counteract this, we added experience replay,
which stored past experiences in memory to be randomly sampled for updates,
helping to improve the stability and efficiency of learning.

```
```
x Memory Constraints:
Managing memory was another consideration due to the experience replay
mechanism, where storing every state-action-reward sequence could become
resource-intensive. We optimized this by periodically pruning less relevant
experiences from the memory buffer, retaining recent and diverse experiences
that helped in fine-tuning the model^
```
```
] Hyperparameter Tuning:
Finding the right hyperparameters, including learning rate, discount factor, and
epsilon decay rate, was challenging. Each hyperparameter had a significant
impact on performance. For instance, a high learning rate led to abrupt policy
shifts, while a lower rate caused excessively slow learning. Adjusting these
values was an iterative process and one of the most time-consuming parts of
development.
```
## Conclusion


```
In developing the AI agent for Flappy Bird using deep reinforcement learning (DRL) and,
specifically, Deep Q-Networks (DQN), I ran into a bunch of challenges that forced me to
rethink and adjust my approach at several points. Balancing exploration and exploitation
was one of the toughest parts—I needed the agent to explore new actions without just
randomly bouncing around or getting too stuck on specific actions that seemed to work
early on. The sparse rewards didn’t help, either, since the agent didn’t get much positive
feedback at first, making the initial learning phase really slow and a bit frustrating.


```
```
To work around this, I tried reward shaping, which gave the agent small rewards for just
staying alive a bit longer. This little hack helped a lot by giving it incremental goals and a
reason to keep "trying." Then there was the non-stationary environment. Pipes kept
moving, and the bird’s own position changed constantly, so I had to use experience replay
and update strategies to keep the learning process stable without completely overhauling
the network every few steps.


```
```
I also ran into a few practical issues—like memory management for the experience replay.
Balancing resources and keeping the right mix of past experiences in memory made a big
difference in how well the agent performed. Then, of course, I had to mess around a lot
with hyperparameters like the learning rate, epsilon decay, and the discount factor. Every
tiny tweak here seemed to change the way the agent acted, so I spent quite a bit of time
just fine-tuning these values.


```
```
This project didn’t just show me what’s possible with DRL but also pointed out some of
its biggest challenges, giving me a real sense of where reinforcement learning can go and
what might need more work in the future.
```

