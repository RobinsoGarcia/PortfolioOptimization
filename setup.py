from setuptools import setup,find_packages
from setuptools.command.install import install

setup(name='portfolio_optimizer',
      version='1',
      description='Portfolio optimization and backtesting',
      url='https://github.com/RobinsoGarcia/portfolio-optimizer',
      author='Robinson Garcia',
      author_email='rlsg.mec@hotmail.com',
      license='MIT',
      #entry_points={'console_scripts':['build_dataset = portfolio_optimizer.__main__:main']},
      scripts=['portfolio_optimizer/scripts/check_allport.py',
      'portfolio_optimizer/scripts/check_returns.py',
      'portfolio_optimizer/scripts/check_rebal.py',
      'portfolio_optimizer/scripts/build_dataset.py'],
      include_package_data=True,
      packages=find_packages(), #['portfolio_optimizer','portfolio_optimizer.portfolio'],
      package_data={'portfolio_optimizer':['stock_data/*'],'portfolio_optimizer':['scripts/*']},
      zip_safe=False,
      #cmdclass={'install':adjustments},
      install_requires=[
          'cvxopt>=1.1.9',
          'numpy>=1.14.0',
          'pandas>-0.22.0'
          'matplotlib',
          'yahoo-finance>=1.4.0',
          'scipy'
      ]
      )
