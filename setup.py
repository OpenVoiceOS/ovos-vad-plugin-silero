#!/usr/bin/env python3
from setuptools import setup

PLUGIN_ENTRY_POINT = 'ovos-vad-plugin-silero = ' \
                     'ovos_vad_plugin_silero:GladosTTSPlugin'
setup(
    name='ovos-vad-plugin-silero',
    version='0.0.1',
    description='silero VAD plugin for OpenVoiceOS',
    url='https://github.com/OpenVoiceOS/ovos-vad-plugin-silero',
    author='JarbasAi',
    author_email='jarbasai@mailfence.com',
    license='Apache-2.0',
    packages=['ovos_vad_plugin_silero'],
    install_requires=["ovos-plugin-manager>=0.0.11a4",
                      "onnxruntime"],
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='mycroft plugin VAD OVOS OpenVoiceOS',
    entry_points={'ovos.plugin.VAD': PLUGIN_ENTRY_POINT}
)
