from setuptools import setup, find_packages

packages = find_packages()
print(packages)
setup(name='Thesis',
      version='1.0',
      description='...',
      author='Pattarawat Chormai',
      author_email='pat.chormai@gmail.com',
      url='https://github.com/heytitle/thesis-designing-recurrent-neural-networks-for-explainability',
      packages=packages
      )
