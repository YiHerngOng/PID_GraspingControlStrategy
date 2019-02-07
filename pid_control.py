#!/usr/bin/env python

'''
Author : Yi Herng Ong
Description : inherit template_match_class to obtain current pose of object and track its features for PID control
Date: Feb 6 2018
'''
import sys, os
import numpy as np
from template_match_class import *
import pdb


class PID():
'''
class of PID controller

'''
	def __init__(self, kp = 1.0, kd = 0.0, ki = 0.0, reference = None):
		self.kp = kp
		self.kd = kd
		self.ki = ki
		self.reference = reference

	def control(self, feedback, dt = 1, reference = None):
		if reference is None:
			pass
		else:
			self.reference = reference
		error = self.reference - feedback	# error between reference and feedback
		error_diff = (error - prev_error) / dt # differentation of errors
		error_accum += error * dt

		# Compute control output
		P = self.kp * error # P term

		I = self.ki * accumulated_error	# D term

		D = self.kd * error_diff # I term

		return P + I + D

	def 
if __name__ == '__main__':
	pass
	# create some sort of loop to loop through using vision feedback regarding object pose / feature