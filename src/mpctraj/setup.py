import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'mpctraj'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nico',
    maintainer_email='n.smith.2@student.rug.nl',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'traj_fs = mpctraj.fscontroller:main',
            'traj_mpc = mpctraj.mpc:main',
            'traj_mpc_sim = mpctraj.mpc_sim:main',
            'pwm_sweep= mpctraj.pwm_sweep:main',
            'pwm_sweep2 = mpctraj.pwm_sweep2:main',
        ],
    },
)