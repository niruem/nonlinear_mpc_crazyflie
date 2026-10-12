# mpc_crazyflie
Code adapted from [here](https://github.com/llanesc/crazyflie_mpc). 

This work uses a Model Predictive Controller on top of crazyflied built in attitude PID to achieve accurate trajectory tracking of preplanned trajectories.

It uses a simplified dynamics model that assumes the attitude PID responds like a first order system to desired attitude commands.

Uses custom Crazyswarm2 cmdVel() function, created following issue [246](https://github.com/IMRCLab/crazyswarm2/issues/246).

For more in-depth explanation, please refer to the report.
