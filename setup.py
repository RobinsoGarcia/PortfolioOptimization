from setuptools import setup,find_packages
from setuptools.command.install import install

class adjustments(install):
    def run(self):
        install.run(self)
        import portfolio_optimizer
        import sys
        import os
        path_port = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])),'scripts')
        #path_port = 'usr/local/lib/python3.6/dist-packages/portfolio_optimizer*'
        os.system('chmod -R 777 /' + path_port)
        os.system('chmod -R 777 /usr/local/bin/build_dataset.py')
        os.system('chmod -R 777 /usr/local/bin/check_allport.py')
        os.system('chmod -R 777 /usr/local/bin/check_rebal.py')
        os.system('chmod -R 777 /usr/local/bin/check_returns.py')
        print('done adjustments')

setup(name='portfolio_optimizer',
      version='0',
      description='Portfolio optimization and backtesting',
      url='https://github.com/RobinsoGarcia/PortfolioOptimization',
      author='Robinson Garcia',
      author_email='rlsg.mec@hotmail.com',
      license='MIT',
      entry_points={'console_scripts':['build_dataset = portfolio_optimizer.__main__:main']},
      include_package_data=True,
      packages=['portfolio_optimizer','portfolio_optimizer.portfolio'],
      package_data={'portfolio_optimizer':['stock_data/*.csv'],'portfolio_optimizer':['scripts/*']},
      zip_safe=False,
      cmdclass={'install':adjustments},
      install_requires=[
          'cvxopt>=1.1.9',
          'numpy>=1.14.0',
          'pandas>-0.22.0'
          'matplotlib>=2.1.2',
          'yahoo-finance>=1.4.0',
          'scipy'
      ]
      )
