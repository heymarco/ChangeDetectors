from setuptools import setup


CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved :: BSD License',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']


setup(name='change-detectors',
      description='A collection of unsupervised change detection algorithms.',
      url='https://github.com/heymarco/ChangeDetectors',
      version="0.0.1",
      license='MIT',
      author='Florian Kalinke, Marco Heyden',
      classifiers=CLASSIFIERS,
      python_requires=">=3.6",
      packages=["detectors"],
      install_requires=["scikit-multiflow", "numpy", "pandas", "tensorflow"],
      extras_require={
        'plots':  ["matplotlib>=2.0.0", "seaborn"]
        }
      )
