from setuptools import setup,find_packages
find_packages()
setup(name='portfolio_optimizer',
      version='0',
      description='Tool for portfolio optimization and backtesting',
      url='https://github.com/RobinsoGarcia/PortfolioOptimization',
      author='Robinson Garcia',
      author_email='rlsg.mec@hotmail.com',
      license='MIT',
      packages=['portfolio_optimizer'],
      zip_safe=False)

install_requires=[
'cvxopt>=1.1.9',
'numpy>=1.14.0',
'pandas>-0.22.0'
'matplotlib>=2.1.2',
'yahoo-finance>=1.4.0'
]
