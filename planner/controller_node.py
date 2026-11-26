"""Connects our code to Gazebo.

Every DT seconds:  read the car state from /car/odom  ->  decide (v, delta)
                   ->  send (v, omega) on /car/cmd_vel
For now (v, delta) is fixed. Later the MPC will decide it.
"""
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from model import L, DT

V = 0.5                    # forward speed [m/s]
DELTA = math.radians(20)   # steering angle [rad]


class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        self.state = None                                                 # (x, y, theta), unknown until the first message
        self.create_subscription(Odometry, '/car/odom', self.on_odom, 10)  # LISTEN: call on_odom for every new state
        self.cmd_pub = self.create_publisher(Twist, '/car/cmd_vel', 10)   # TALK: channel to send commands on
        self.create_timer(DT, self.on_timer)                              # call on_timer every DT = 0.1 s

    def on_odom(self, msg):
        """Store the latest state. Heading: quaternion -> angle theta."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        theta = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
        self.state = (p.x, p.y, theta)

    def on_timer(self):
        """One control step."""
        if self.state is None:
            return                                   # no state from Gazebo yet
        x, y, theta = self.state

        v, delta = V, DELTA                          # LATER: v, delta = mpc(...)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = v / L * math.tan(delta)      # omega = theta_dot from the bicycle model
        self.cmd_pub.publish(cmd)
        self.get_logger().info(f'x = {x:5.2f}  y = {y:5.2f}  theta = {math.degrees(theta):7.1f} deg')


def main():
    rclpy.init()
    rclpy.spin(Controller())


if __name__ == '__main__':
    main()
