from distutils.core import setup

setup(
    name='frgtools',
    version='0.1dev',
    packages=['frgtools'],
    install_requires = [
    	'matplotlib-scalebar',
    	'imreg_dft'
    	],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('./frgtools/README.txt').read(),
)