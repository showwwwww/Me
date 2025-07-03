# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Author  :   一力辉 
'''

from .difyAgent import DifyApiAgent
from .repeaterAgent import RepeaterAgent
from .fastgptAgent import FastgptApiAgent
from .openaiAgent import OpenaiApiAgent
from .outsideAgent import OutsideAgent

from .agentFactory import AgentFactory

__all__ = ['AgentFactory']