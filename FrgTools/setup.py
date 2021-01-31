from distutils.core import setup

setup(
    name='frgtools',
    version='0.1dev',
    packages=['frgtools'],
    author = 'Rishi Kumar',
    author_email = 'rishikumar@ucsd.edu',
    install_requires = [
    	'matplotlib-scalebar',
    	'imreg_dft',
        'renishawWiRE',
        'scipy',
        'matplotlib',
        'numpy',
        'opencv-python',
        'tqdm',
        'skimage',
        'affine6p',
        'smtplib'
    	],
    license='MIT',
    long_description=open('./frgtools/README.txt').read(),
)