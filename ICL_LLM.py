import numpy as np
import openai
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, ExpOpt, Opt
from tqdm import *
from zai import ZhipuAiClient


def CorrectionShift_ICL(num_shots,test_num, X_train, y_train, X_test, y_test, MODEL_NAME, API_KEY, BASE_URL):

	demos = []
	prompt = ''
	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'


	for i in range(0,num_shots):
		demos.append((f'Feature 0: {X_test['duration'][i]}\n'+
		f'Feature 1: {X_test['amount'][i]}\n'+
		f'Feature 2: {X_test['age'][i]}\n'+
		f'Feature 3: {X_test['personal_status_sex'][i]}\n'+
		f'Output: {y_test[i]}\n\n'))
		prompt += demos[i]

	client = ZhipuAiClient(api_key=API_KEY)
	for i in range(0, test_num):

		query = (f'Feature 0: {X_train['duration'][i]}\n'+
		f'Feature 1: {X_train['amount'][i]}\n'+
		f'Feature 2: {X_train['age'][i]}\n'+
		f'Feature 3: {X_train['personal_status_sex'][i]}\n'+
		f'Output: ')
		text = instruction + prompt + query

		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text},
			],
			thinking={
				"type": "disabled",  # 启用深度思考模式
			},
			stream=False,  # 启用流式输出
			temperature=0  # 控制输出的随机性
		)
		predicted = response.choices[0].message.content
		print(f'第{i}个查询：{predicted}')
		if predicted == '1':
			predicted = 1
		elif predicted == '0':
			predicted = 0
		else:
			predicted = 2222
		prediction_target.append(predicted)
	acc = np.mean(prediction_target == y_train.values[0:test_num])

	return acc, np.array(prediction_target), prompt




def CorrectionShift_ICL_recourse_validity(X_recourse, target, prompt, MODEL_NAME, API_KEY, BASE_URL):

	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'


	client = ZhipuAiClient(api_key=API_KEY)

	for i in range(0, X_recourse.shape[0]):
		query = (f'Feature 0: {X_recourse[i][0]}\n' +
		f'Feature 1: {X_recourse[i][1]}\n' +
		f'Feature 2: {X_recourse[i][2]}\n' +
		f'Feature 3: {X_recourse[i][3]}\n' +
		f'Output: ')
		text = instruction + prompt + query

		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text},
			],
			thinking={
				"type": "disabled",  # 启用深度思考模式
			},
			stream=False,  # 启用流式输出
			temperature=0  # 控制输出的随机性
		)
		predicted = response.choices[0].message.content
		if predicted == '1':
			predicted = 1
		elif predicted == '0':
			predicted = 0
		else:
			predicted = 2222
		prediction_target.append(predicted)
	validity = np.mean(prediction_target == np.array([target]*X_recourse.shape[0]))

	return validity


def CorrectionShift_recourse(X_old,mean,var,prompt,MODEL_NAME,API_KEY,BASE_URL,lamb=0):
	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'
	predicted_recourse = []
	loss = []
	client = ZhipuAiClient(api_key=API_KEY)

	for i in range(X_old.shape[0]):

		def ackley(solution):
			x = solution.get_x()
			query = (f'Feature 0: {x[0]}\n' +
			f'Feature 1: {x[1]}\n' +
			f'Feature 2: {x[2]}\n' +
			f'Feature 3: {x[3]}\n' +
			f'Output: ')
			text = instruction + prompt + query


			response = client.chat.completions.create(
				model=MODEL_NAME,
				messages=[
					{"role": "user", "content": text},
				],
				thinking={
					"type": "disabled",  # 启用深度思考模式
				},
				stream=False,  # 启用流式输出
				temperature=0  # 控制输出的随机性
			)
			predicted = response.choices[0].message.content
			if predicted == '1':
				predicted = 1.0
			elif predicted == '0':
				predicted = 0.0
			else:
				predicted = 2222.0
			x_old = np.array([X_old[i][0], X_old[i][1], X_old[i][3]]).astype(float)
			x_new = np.array([x[0], x[1], x[2]]).astype(float)
			num_feature = ["duration", "amount", "age"]
			for index,feature in enumerate(num_feature):
				x_old[index] = (x_old[index] - mean[feature]) / var[feature]
				x_new[index] = (x_new[index] - mean[feature]) / var[feature]
			cost = x_new - x_old
			loss = np.linalg.norm(np.array([predicted-1.0]),1) + lamb * np.linalg.norm(cost,1)

			return loss

		dim_list = [
			(ValueType.DISCRETE, [0, 80], False),
			(ValueType.DISCRETE, [250, 19000], False),
			(ValueType.DISCRETE, [X_old[i][3], X_old[i][3]+10], False),
			(ValueType.DISCRETE, [X_old[i][2], X_old[i][2]], False),
		]

		dim = Dimension2(dim_list)

		objective = Objective(ackley, dim)
		parameter = Parameter(budget=1000)

		sol = Opt.min(objective, parameter)
		answer = sol.get_x()
		predicted_recourse.append(answer)
		loss.append(sol.get_value())
	return np.array(predicted_recourse), np.array(loss)


def CorrectionShift_choose_lambda(X_old, target, mean, var, prompt, MODEL_NAME, API_KEY, BASE_URL):
	# lambdas = np.arange(0.1, 1.1, 0.1)
	lambdas = [0.1]

	v_old = 0
	recourse_history = []
	validity_history = []
	loss = []
	for i, lamb in tqdm(enumerate(lambdas)):
		print('测试lamda中')
		print("Testing lambda:%f" % lamb)
		r,l = CorrectionShift_recourse(X_old, mean=mean, var=var, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL, lamb=lamb)
		v = CorrectionShift_ICL_recourse_validity(X_recourse=r, target=target, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL)
		recourse_history.append(r)
		validity_history.append(v)
		loss.append(l)
		print(f'lamb:{lamb}, validity:{v}, loss:{l}')
		if v > v_old:
			v_old = v
		else:
			li = max(0, i - 1)
			print(f'search end,the finding is {lambdas[li]}')
			return lambdas[li], recourse_history[li], validity_history[li], np.mean(loss[li])
	print(f'search end,the finding is {lamb}')
	return lamb, recourse_history[-1], validity_history[-1], np.mean(loss[-1])





#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL
#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL
#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL
#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL
#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL
#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL#TemporalShift_ICL


def TemporalShift_ICL(num_shots,test_num, X_train, y_train, X_test, y_test, MODEL_NAME, API_KEY, BASE_URL):

	demos = []
	prompt = ''
	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'


	for i in range(0,num_shots):
		demos.append((f'Feature 0: {X_test['Zip'][i]}\n'+
		f'Feature 1: {X_test['NAICS'][i]}\n'+
		f'Feature 2: {X_test['ApprovalDate'][i]}\n'+
		f'Feature 3: {X_test['ApprovalFY'][i]}\n'+
		f'Feature 4: {X_test['Term'][i]}\n'+
		f'Feature 5: {X_test['NoEmp'][i]}\n'+
		f'Feature 6: {X_test['NewExist'][i]}\n'+
		f'Feature 7: {X_test['CreateJob'][i]}\n'+
		f'Feature 8: {X_test['RetainedJob'][i]}\n'+
		f'Feature 9: {X_test['FranchiseCode'][i]}\n'+
		f'Feature 10: {X_test['UrbanRural'][i]}\n'+
		f'Feature 11: {X_test['RevLineCr'][i]}\n' +
		f'Feature 12: {X_test['ChgOffDate'][i]}\n'+
		f'Feature 13: {X_test['DisbursementDate'][i]}\n'+
		f'Feature 14: {X_test['DisbursementGross'][i]}\n'+
		f'Feature 15: {X_test['ChgOffPrinGr'][i]}\n'+
		f'Feature 16: {X_test['GrAppv'][i]}\n'+
		f'Feature 17: {X_test['SBA_Appv'][i]}\n'+
		f'Feature 18: {X_test['New'][i]}\n'+
		f'Feature 19: {X_test['RealEstate'][i]}\n'+
		f'Feature 20: {X_test['Portion'][i]}\n'+
		f'Feature 21: {X_test['Recession'][i]}\n'+
		f'Feature 22: {X_test['daysterm'][i]}\n'+
		f'Feature 23: {X_test['xx'][i]}\n'+
		f'Output: {y_test[i]}\n\n'))
		prompt += demos[i]

	client = ZhipuAiClient(api_key=API_KEY)

	for i in range(0, test_num):
		query = (f'Feature 0: {X_train['Zip'][i]}\n'+
		f'Feature 1: {X_train['NAICS'][i]}\n'+
		f'Feature 2: {X_train['ApprovalDate'][i]}\n'+
		f'Feature 3: {X_train['ApprovalFY'][i]}\n'+
		f'Feature 4: {X_train['Term'][i]}\n'+
		f'Feature 5: {X_train['NoEmp'][i]}\n'+
		f'Feature 6: {X_train['NewExist'][i]}\n'+
		f'Feature 7: {X_train['CreateJob'][i]}\n'+
		f'Feature 8: {X_train['RetainedJob'][i]}\n'+
		f'Feature 9: {X_train['FranchiseCode'][i]}\n'+
		f'Feature 10: {X_train['UrbanRural'][i]}\n'+
		f'Feature 11: {X_train['RevLineCr'][i]}\n' +
		f'Feature 12: {X_train['ChgOffDate'][i]}\n'+
		f'Feature 13: {X_train['DisbursementDate'][i]}\n'+
		f'Feature 14: {X_train['DisbursementGross'][i]}\n'+
		f'Feature 15: {X_train['ChgOffPrinGr'][i]}\n'+
		f'Feature 16: {X_train['GrAppv'][i]}\n'+
		f'Feature 17: {X_train['SBA_Appv'][i]}\n'+
		f'Feature 18: {X_train['New'][i]}\n'+
		f'Feature 19: {X_train['RealEstate'][i]}\n'+
		f'Feature 20: {X_train['Portion'][i]}\n'+
		f'Feature 21: {X_train['Recession'][i]}\n'+
		f'Feature 22: {X_train['daysterm'][i]}\n'+
		f'Feature 23: {X_train['xx'][i]}\n'+
		f'Output: ')

		text = instruction + prompt + query

		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text},
			],
			thinking={
				"type": "disabled",  # 启用深度思考模式
			},
			stream=False,  # 启用流式输出
			temperature=0,
		)
		predicted = response.choices[0].message.content
		print(f'第{i}个查询：{predicted}')
		if predicted == '1':
			predicted = 1
		elif predicted == '0':
			predicted = 0
		else:
			predicted = 2222
		prediction_target.append(predicted)
	acc = np.mean(prediction_target == y_train.values[0:test_num])

	return acc, np.array(prediction_target), prompt


def TemporalShift_ICL_recourse_validity(X_recourse, target, prompt, MODEL_NAME, API_KEY, BASE_URL):

	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'

	client = ZhipuAiClient(api_key=API_KEY)

	for i in range(0, X_recourse.shape[0]):
		query = (f'Feature 0: {X_recourse[i][0]}\n'+
		f'Feature 1: {X_recourse[i][1]}\n'+
		f'Feature 2: {X_recourse[i][2]}\n'+
		f'Feature 3: {X_recourse[i][3]}\n'+
		f'Feature 4: {X_recourse[i][4]}\n'+
		f'Feature 5: {X_recourse[i][5]}\n'+
		f'Feature 6: {X_recourse[i][6]}\n'+
		f'Feature 7: {X_recourse[i][7]}\n'+
		f'Feature 8: {X_recourse[i][8]}\n'+
		f'Feature 9: {X_recourse[i][9]}\n'+
		f'Feature 10: {X_recourse[i][10]}\n'+
		f'Feature 11: {X_recourse[i][11]}\n'+
		f'Feature 12: {X_recourse[i][12]}\n'+
		f'Feature 13: {X_recourse[i][13]}\n'+
		f'Feature 14: {X_recourse[i][14]}\n'+
		f'Feature 15: {X_recourse[i][15]}\n'+
		f'Feature 16: {X_recourse[i][16]}\n'+
		f'Feature 17: {X_recourse[i][17]}\n'+
		f'Feature 18: {X_recourse[i][18]}\n'+
		f'Feature 19: {X_recourse[i][19]}\n'+
		f'Feature 20: {X_recourse[i][20]}\n'+
		f'Feature 21: {X_recourse[i][21]}\n'+
		f'Feature 22: {X_recourse[i][22]}\n'+
		f'Feature 23: {X_recourse[i][23]}\n'+
		f'Output: ')

		text = instruction + prompt + query

		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text},
			],
			thinking={
				"type": "disabled",  # 启用深度思考模式
			},
			stream=False,  # 启用流式输出
			temperature=0,
		)
		predicted = response.choices[0].message.content
		if predicted == '1':
			predicted = 1
		elif predicted == '0':
			predicted = 0
		else:
			predicted = 2222
		prediction_target.append(predicted)
	validity = np.mean(prediction_target == np.array([target]*X_recourse.shape[0]))

	return validity






def TemporalShift_recourse(X_old,mean,var,prompt,MODEL_NAME,API_KEY,BASE_URL,lamb=0):
	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'
	predicted_recourse = []
	loss = []

	client = ZhipuAiClient(api_key=API_KEY)

	for i in range(X_old.shape[0]):

		def ackley(solution):
			x = solution.get_x()
			query = (f'Feature 0: {x[0]}\n' +
			f'Feature 1: {x[1]}\n' +
			f'Feature 2: {x[2]}\n' +
			f'Feature 3: {x[3]}\n' +
			f'Feature 4: {x[4]}\n' +
			f'Feature 5: {x[5]}\n' +
			f'Feature 6: {x[6]}\n' +
			f'Feature 7: {x[7]}\n' +
			f'Feature 8: {x[8]}\n' +
			f'Feature 9: {x[9]}\n' +
			f'Feature 10: {x[10]}\n' +
			f'Feature 11: {x[11]}\n' +
			f'Feature 12: {x[12]}\n' +
			f'Feature 13: {x[13]}\n' +
			f'Feature 14: {x[14]}\n' +
			f'Feature 15: {x[15]}\n' +
			f'Feature 16: {x[16]}\n' +
			f'Feature 17: {x[17]}\n' +
			f'Feature 18: {x[18]}\n' +
			f'Feature 19: {x[19]}\n' +
			f'Feature 20: {x[20]}\n' +
			f'Feature 21: {x[21]}\n' +
			f'Feature 22: {x[22]}\n' +
			f'Feature 23: {x[23]}\n' +
			f'Output: ')

			text = instruction + prompt + query


			response = client.chat.completions.create(
				model=MODEL_NAME,
				messages=[
					{"role": "user", "content": text},
				],
				thinking={
					"type": "disabled",  # 启用深度思考模式
				},
				stream=False,  # 启用流式输出
				temperature=0,
			)
			predicted = response.choices[0].message.content
			if predicted == '1':
				predicted = 1.0
			elif predicted == '0':
				predicted = 0.0
			else:
				predicted = 2222
			x_old = np.array([X_old[i][0], X_old[i][1], X_old[i][2],X_old[i][3], X_old[i][4], X_old[i][5],X_old[i][6], X_old[i][7], X_old[i][8],X_old[i][9], X_old[i][10],X_old[i][11],X_old[i][12], X_old[i][13], X_old[i][14],X_old[i][15], X_old[i][16], X_old[i][17],X_old[i][18], X_old[i][19], X_old[i][20],X_old[i][21], X_old[i][22], X_old[i][23]]).astype(float)
			x_new = np.array([x[0], x[1], x[2],x[3], x[4], x[5],x[6], x[7], x[8],x[9],x[10],x[11],x[12], x[13], x[14],x[15], x[16], x[17],x[18], x[19], x[20],x[21], x[22], x[23]]).astype(float)
			all_feature = ['Zip', 'NAICS', 'ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp', 'NewExist', 'CreateJob',
						'RetainedJob', 'FranchiseCode', 'UrbanRural','RevLineCr', 'ChgOffDate', 'DisbursementDate',
						'DisbursementGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'New', 'RealEstate', 'Portion',
						'Recession', 'daysterm', 'xx']
			for index,feature in enumerate(all_feature):
				x_old[index] = (x_old[index] - mean[feature]) / var[feature]
				x_new[index] = (x_new[index] - mean[feature]) / var[feature]
			cost = x_new - x_old
			loss = np.linalg.norm(np.array([predicted-1.0]),1) + lamb * np.linalg.norm(cost,1)

			return loss

		dim_list = [
			(ValueType.DISCRETE, [X_old[i][0], X_old[i][0]], False),
			(ValueType.DISCRETE, [X_old[i][1] , X_old[i][1]], False),
			(ValueType.DISCRETE, [X_old[i][2], X_old[i][2]], False),
			(ValueType.DISCRETE, [X_old[i][3], X_old[i][3]], False),

			(ValueType.DISCRETE, [0, 350], False),
			(ValueType.DISCRETE, [0, 650], False),
			(ValueType.GRID, [0, 1, 2]),
			(ValueType.DISCRETE, [0, 130], False),
			(ValueType.DISCRETE, [0, 550], False),
			(ValueType.DISCRETE, [0, 90000], False),

			(ValueType.GRID, [0, 1, 2]),

			(ValueType.GRID, [0, 1, 2, 3]),

			(ValueType.DISCRETE, [X_old[i][12], X_old[i][12]], False),
			(ValueType.DISCRETE, [X_old[i][13], X_old[i][13]], False),
			(ValueType.DISCRETE, [X_old[i][14]-1000, X_old[i][14] + 5000], False),
			(ValueType.DISCRETE, [X_old[i][15] - 1000, X_old[i][15] + 5000], False),
			(ValueType.DISCRETE, [X_old[i][16] - 1000, X_old[i][16] + 5000], False),
			(ValueType.DISCRETE, [X_old[i][17] - 1000, X_old[i][17] + 5000], False),

			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.DISCRETE, [X_old[i][22] - 500, X_old[i][22] + 500], False),
			(ValueType.DISCRETE, [X_old[i][23] - 1000, X_old[i][23] + 5000], False),
		]

		dim = Dimension2(dim_list)

		objective = Objective(ackley, dim)
		parameter = Parameter(budget=1000)

		sol = Opt.min(objective, parameter)
		answer = sol.get_x()
		predicted_recourse.append(answer)
		loss.append(sol.get_value())
	return np.array(predicted_recourse), np.array(loss)



def TemporalShift_choose_lambda(X_old, target, mean, var, prompt, MODEL_NAME, API_KEY, BASE_URL):
	lambdas = np.arange(0.1, 1.1, 0.1)
	lambdas = [0.1]

	v_old = 0
	recourse_history = []
	validity_history = []
	loss = []
	for i, lamb in tqdm(enumerate(lambdas)):
		print('测试lamda中')
		print("Testing lambda:%f" % lamb)
		r,l = TemporalShift_recourse(X_old, mean=mean, var=var, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL, lamb=lamb)
		v = TemporalShift_ICL_recourse_validity(X_recourse=r, target=target, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL)
		print(l)
		recourse_history.append(r)
		validity_history.append(v)
		loss.append(l)
		print(f'lamb:{lamb}, validity:{v}, loss:{l}')
		if v > v_old:
			v_old = v
		else:
			li = max(0, i - 1)
			print(f'search end,the finding is {lambdas[li]}')
			return lambdas[li], recourse_history[li], validity_history[li], np.mean(loss[li])
	print(f'search end,the finding is {lamb}')
	return lamb, recourse_history[-1], validity_history[-1], np.mean(loss[-1])






#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL
#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL
#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL
#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL
#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL
#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL#GeospatialShift_ICL



def GeospatialShift_ICL(num_shots,test_num, X_train, y_train, X_test, y_test, MODEL_NAME, API_KEY, BASE_URL):

	demos = []
	prompt = ''
	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'


	for i in range(0,num_shots):
		demos.append((f'Feature 0: {X_test['sex'][i]}\n'+
		f'Feature 1: {X_test['age'][i]}\n'+
		f'Feature 2: {X_test['address'][i]}\n'+
		f'Feature 3: {X_test['famsize'][i]}\n'+
		f'Feature 4: {X_test['Pstatus'][i]}\n'+
		f'Feature 5: {X_test['Medu'][i]}\n'+
		f'Feature 6: {X_test['Fedu'][i]}\n'+
		f'Feature 7: {X_test['Mjob'][i]}\n'+
		f'Feature 8: {X_test['Fjob'][i]}\n'+
		f'Feature 9: {X_test['reason'][i]}\n'+
		f'Feature 10: {X_test['guardian'][i]}\n'+
		f'Feature 11: {X_test['traveltime'][i]}\n' +
		f'Feature 12: {X_test['studytime'][i]}\n'+
		f'Feature 13: {X_test['failures'][i]}\n'+
		f'Feature 14: {X_test['schoolsup'][i]}\n'+
		f'Feature 15: {X_test['famsup'][i]}\n'+
		f'Feature 16: {X_test['paid'][i]}\n'+
		f'Feature 17: {X_test['activities'][i]}\n'+
		f'Feature 18: {X_test['nursery'][i]}\n'+
		f'Feature 19: {X_test['higher'][i]}\n'+
		f'Feature 20: {X_test['internet'][i]}\n'+
		f'Feature 21: {X_test['romantic'][i]}\n'+
		f'Feature 22: {X_test['famrel'][i]}\n'+
		f'Feature 23: {X_test['freetime'][i]}\n'+
		f'Feature 24: {X_test['goout'][i]}\n'+
		f'Feature 25: {X_test['Dalc'][i]}\n'+
		f'Feature 26: {X_test['Walc'][i]}\n'+
		f'Feature 27: {X_test['health'][i]}\n'+
		f'Feature 28: {X_test['absences'][i]}\n'+
		f'Output: {y_test[i]}\n\n'))
		prompt += demos[i]

	client = ZhipuAiClient(api_key=API_KEY)  # 请填写您自己的 API Key

	for i in range(0, test_num):
		query = (f'Feature 0: {X_train['sex'][i]}\n'+
		f'Feature 1: {X_train['age'][i]}\n'+
		f'Feature 2: {X_train['address'][i]}\n'+
		f'Feature 3: {X_train['famsize'][i]}\n'+
		f'Feature 4: {X_train['Pstatus'][i]}\n'+
		f'Feature 5: {X_train['Medu'][i]}\n'+
		f'Feature 6: {X_train['Fedu'][i]}\n'+
		f'Feature 7: {X_train['Mjob'][i]}\n'+
		f'Feature 8: {X_train['Fjob'][i]}\n'+
		f'Feature 9: {X_train['reason'][i]}\n'+
		f'Feature 10: {X_train['guardian'][i]}\n'+
		f'Feature 11: {X_train['traveltime'][i]}\n' +
		f'Feature 12: {X_train['studytime'][i]}\n'+
		f'Feature 13: {X_train['failures'][i]}\n'+
		f'Feature 14: {X_train['schoolsup'][i]}\n'+
		f'Feature 15: {X_train['famsup'][i]}\n'+
		f'Feature 16: {X_train['paid'][i]}\n'+
		f'Feature 17: {X_train['activities'][i]}\n'+
		f'Feature 18: {X_train['nursery'][i]}\n'+
		f'Feature 19: {X_train['higher'][i]}\n'+
		f'Feature 20: {X_train['internet'][i]}\n'+
		f'Feature 21: {X_train['romantic'][i]}\n'+
		f'Feature 22: {X_train['famrel'][i]}\n'+
		f'Feature 23: {X_train['freetime'][i]}\n'+
		f'Feature 24: {X_train['goout'][i]}\n'+
		f'Feature 25: {X_train['Dalc'][i]}\n'+
		f'Feature 26: {X_train['Walc'][i]}\n'+
		f'Feature 27: {X_train['health'][i]}\n'+
		f'Feature 28: {X_train['absences'][i]}\n'+
		f'Output: ')

		text = instruction + prompt + query
		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text},
			],
			thinking={
				"type": "disabled",  # 启用深度思考模式
			},
			stream=False,  # 启用流式输出
			temperature=0  # 控制输出的随机性
		)
		predicted = response.choices[0].message.content

		print(f'第{i}个查询：{predicted}')
		if predicted == '1':
			predicted = 1
		elif predicted == '0':
			predicted = 0
		else:
			predicted = 2222
		prediction_target.append(predicted)
	acc = np.mean(prediction_target == y_train.values[0:test_num])

	return acc, np.array(prediction_target), prompt



def GeospatialShift_ICL_recourse_validity(X_recourse, target, prompt, MODEL_NAME, API_KEY, BASE_URL):

	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'

	client = ZhipuAiClient(api_key=API_KEY)  # 请填写您自己的 API Key

	for i in range(0, X_recourse.shape[0]):
		query = (f'Feature 0: {X_recourse[i][0]}\n'+
		f'Feature 1: {X_recourse[i][1]}\n'+
		f'Feature 2: {X_recourse[i][2]}\n'+
		f'Feature 3: {X_recourse[i][3]}\n'+
		f'Feature 4: {X_recourse[i][4]}\n'+
		f'Feature 5: {X_recourse[i][5]}\n'+
		f'Feature 6: {X_recourse[i][6]}\n'+
		f'Feature 7: {X_recourse[i][7]}\n'+
		f'Feature 8: {X_recourse[i][8]}\n'+
		f'Feature 9: {X_recourse[i][9]}\n'+
		f'Feature 10: {X_recourse[i][10]}\n'+
		f'Feature 11: {X_recourse[i][11]}\n'+
		f'Feature 12: {X_recourse[i][12]}\n'+
		f'Feature 13: {X_recourse[i][13]}\n'+
		f'Feature 14: {X_recourse[i][14]}\n'+
		f'Feature 15: {X_recourse[i][15]}\n'+
		f'Feature 16: {X_recourse[i][16]}\n'+
		f'Feature 17: {X_recourse[i][17]}\n'+
		f'Feature 18: {X_recourse[i][18]}\n'+
		f'Feature 19: {X_recourse[i][19]}\n'+
		f'Feature 20: {X_recourse[i][20]}\n'+
		f'Feature 21: {X_recourse[i][21]}\n'+
		f'Feature 22: {X_recourse[i][22]}\n'+
		f'Feature 23: {X_recourse[i][23]}\n'+
		f'Feature 24: {X_recourse[i][23]}\n'+
		f'Feature 25: {X_recourse[i][23]}\n'+
		f'Feature 26: {X_recourse[i][23]}\n'+
		f'Feature 27: {X_recourse[i][23]}\n'+
		f'Feature 28: {X_recourse[i][23]}\n'+
		f'Output: ')

		text = instruction + prompt + query
		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text},
			],
			thinking={
				"type": "disabled",  # 启用深度思考模式
			},
			stream=False,  # 启用流式输出
			temperature=0  # 控制输出的随机性
		)
		predicted = response.choices[0].message.content
		if predicted == '1':
			predicted = 1
		elif predicted == '0':
			predicted = 0
		else:
			predicted = 2222
		prediction_target.append(predicted)
	validity = np.mean(prediction_target == np.array([target]*X_recourse.shape[0]))

	return validity






def GeospatialShift_recourse(X_old,mean,var,prompt,MODEL_NAME,API_KEY,BASE_URL,lamb=0):
	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'
	predicted_recourse = []
	loss = []
	client = ZhipuAiClient(api_key=API_KEY)  # 请填写您自己的 API Key
	for i in range(X_old.shape[0]):

		def ackley(solution):
			x = solution.get_x()
			query = (f'Feature 0: {x[0]}\n' +
			f'Feature 1: {x[1]}\n' +
			f'Feature 2: {x[2]}\n' +
			f'Feature 3: {x[3]}\n' +
			f'Feature 4: {x[4]}\n' +
			f'Feature 5: {x[5]}\n' +
			f'Feature 6: {x[6]}\n' +
			f'Feature 7: {x[7]}\n' +
			f'Feature 8: {x[8]}\n' +
			f'Feature 9: {x[9]}\n' +
			f'Feature 10: {x[10]}\n' +
			f'Feature 11: {x[11]}\n' +
			f'Feature 12: {x[12]}\n' +
			f'Feature 13: {x[13]}\n' +
			f'Feature 14: {x[14]}\n' +
			f'Feature 15: {x[15]}\n' +
			f'Feature 16: {x[16]}\n' +
			f'Feature 17: {x[17]}\n' +
			f'Feature 18: {x[18]}\n' +
			f'Feature 19: {x[19]}\n' +
			f'Feature 20: {x[20]}\n' +
			f'Feature 21: {x[21]}\n' +
			f'Feature 22: {x[22]}\n' +
			f'Feature 23: {x[23]}\n' +
			f'Feature 24: {x[24]}\n' +
			f'Feature 25: {x[25]}\n' +
			f'Feature 26: {x[26]}\n' +
			f'Feature 27: {x[27]}\n' +
			f'Feature 28: {x[28]}\n' +
			f'Output: ')

			text = instruction + prompt + query

			response = client.chat.completions.create(
				model=MODEL_NAME,
				messages=[
					{"role": "user", "content": text},
				],
				thinking={
					"type": "disabled",  # 启用深度思考模式
				},
				stream=False,  # 启用流式输出
				temperature=0  # 控制输出的随机性
			)
			predicted = response.choices[0].message.content
			if predicted == '1':
				predicted = 1.0
			elif predicted == '0':
				predicted = 0.0
			else:
				predicted = 2222
			x_old = np.array([X_old[i][0], X_old[i][1], X_old[i][2],X_old[i][3], X_old[i][4], X_old[i][5],X_old[i][6], X_old[i][7], X_old[i][8],X_old[i][9], X_old[i][10],X_old[i][11],X_old[i][12], X_old[i][13], X_old[i][14],X_old[i][15], X_old[i][16], X_old[i][17],X_old[i][18], X_old[i][19], X_old[i][20],X_old[i][21], X_old[i][22], X_old[i][23], X_old[i][24], X_old[i][25], X_old[i][26], X_old[i][27], X_old[i][28]]).astype(float)
			x_new = np.array([x[0], x[1], x[2],x[3], x[4], x[5],x[6], x[7], x[8],x[9], x[10],x[11],x[12], x[13], x[14],x[15], x[16], x[17],x[18], x[19], x[20],x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28]]).astype(float)
			all_feature = ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
						   'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
						   'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
						   'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
						   'health', 'absences']
			for index,feature in enumerate(all_feature):
				x_old[index] = (x_old[index] - mean[feature]) / var[feature]
				x_new[index] = (x_new[index] - mean[feature]) / var[feature]
			cost = x_new - x_old
			loss = np.linalg.norm(np.array([predicted-1.0]),1) + lamb * np.linalg.norm(cost,1)

			return loss

		dim_list = [
			(ValueType.DISCRETE, [X_old[i][0] , X_old[i][0]], False),
			(ValueType.DISCRETE, [X_old[i][1] , X_old[i][1]+10], False),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
            (ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1, 2, 3, 4]),
			(ValueType.GRID, [0, 1, 2, 3, 4]),
			(ValueType.GRID, [0, 1, 2, 3, 4]),
			(ValueType.GRID, [0, 1, 2, 3, 4]),
			(ValueType.GRID, [0, 1, 2, 3]),
			(ValueType.GRID, [0, 1, 2]),
            (ValueType.GRID, [1, 2, 3, 4]),
            (ValueType.GRID, [1, 2, 3, 4]),
			(ValueType.GRID, [0, 1, 2, 3]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.GRID, [0, 1]),
			(ValueType.DISCRETE, [1, 5], False),
			(ValueType.DISCRETE, [1, 5], False),
			(ValueType.DISCRETE, [1, 5], False),
			(ValueType.DISCRETE, [1, 5], False),
			(ValueType.DISCRETE, [1, 5], False),
			(ValueType.DISCRETE, [1, 5], False),
			(ValueType.DISCRETE, [0, 10], False),
		]

		dim = Dimension2(dim_list)

		objective = Objective(ackley, dim)
		parameter = Parameter(budget=1000)
		sol = Opt.min(objective, parameter)
		answer = sol.get_x()
		predicted_recourse.append(answer)
		loss.append(sol.get_value())
	return np.array(predicted_recourse), np.array(loss)



def GeospatialShift_choose_lambda(X_old, target, mean, var, prompt, MODEL_NAME, API_KEY, BASE_URL):
	# lambdas = np.arange(0.1, 1.1, 0.1)
	lambdas = [0.1]

	v_old = 0
	recourse_history = []
	validity_history = []
	loss = []
	for i, lamb in tqdm(enumerate(lambdas)):
		print('测试lamda中')
		print("Testing lambda:%f" % lamb)
		r,l = GeospatialShift_recourse(X_old, mean=mean, var=var, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL, lamb=lamb)
		v = GeospatialShift_ICL_recourse_validity(X_recourse=r, target=target, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL)
		recourse_history.append(r)
		validity_history.append(v)
		loss.append(l)
		print(f'lamb:{lamb}, validity:{v}, loss:{l}')
		if v > v_old:
			v_old = v
		else:
			li = max(0, i - 1)
			print(f'search end,the finding is {lambdas[li]}')
			return lambdas[li], recourse_history[li], validity_history[li], np.mean(loss[li])
	print(f'search end,the finding is {lamb}')
	return lamb, recourse_history[-1], validity_history[-1], np.mean(loss[-1])



#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL
#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL
#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL
#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL
#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL#SimulatedData_ICL




def SimulatedData_ICL(num_shots,test_num, X_train, y_train, X_test, y_test, MODEL_NAME, API_KEY, BASE_URL):

	demos = []
	prompt = ''
	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'


	for i in range(0,num_shots):
		demos.append((f'Feature 0: {X_test['X0'][i]}\n'+
		f'Feature 1: {X_test['X1'][i]}\n'+
		f'Output: {y_test[i]}\n\n'))
		prompt += demos[i]
	# print(prompt)
	client = openai.OpenAI(
		api_key=API_KEY,  # 请替换为你的实际 API Key
		base_url=BASE_URL  # 替换为你的实际 base_url
	)

	# for i in range(0,X_train.index[-1]+1):
	for i in range(0, test_num):
		query = (f'Feature 0: {X_train['X0'][i]}\n'+
		f'Feature 1: {X_train['X1'][i]}\n'+
		f'Output: ')

		text = instruction + prompt + query
		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text}
			],
			# max_tokens=10,
            extra_body={'do_sample': False}
        )
		predicted = response.choices[0].message.content

		print(f'第{i}个查询：{predicted}')
		if predicted == '1':
			predicted = 1
		if predicted == '0':
			predicted = 0
		prediction_target.append(predicted)
	acc = np.mean(prediction_target == y_train.values[0:test_num])

	return acc, np.array(prediction_target), prompt



def SimulatedData_ICL_recourse_validity(X_recourse, target, prompt, MODEL_NAME, API_KEY, BASE_URL):

	prediction_target = []

	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'

	client = openai.OpenAI(
		api_key=API_KEY,  # 请替换为你的实际 API Key
		base_url=BASE_URL  # 替换为你的实际 base_url
	)
	for i in range(0, X_recourse.shape[0]):
		query = (f'Feature 0: {X_recourse[i][0]}\n'+
		f'Feature 1: {X_recourse[i][1]}\n'+
		f'Output: ')

		text = instruction + prompt + query
		response = client.chat.completions.create(
			model=MODEL_NAME,
			messages=[
				{"role": "user", "content": text}
			],
			temperature=0,
			extra_body={'do_sample': False}

		)
		predicted = response.choices[0].message.content
		if predicted == '1':
			predicted = 1
		if predicted == '0':
			predicted = 0
		prediction_target.append(predicted)
	validity = np.mean(prediction_target == np.array([target]*X_recourse.shape[0]))

	return validity



def SimulatedData_recourse(X_old,prompt,target,MODEL_NAME,API_KEY,BASE_URL,lamb=0):
	instruction = 'The task is to provide your best estimate for ”Output”. Please provide that and only that, without any additional text.\n\n\n'
	predicted_recourse = []
	loss = []
	client = openai.OpenAI(
		api_key=API_KEY,  # 请替换为你的实际 API Key
		base_url=BASE_URL  # 替换为你的实际 base_url
	)
	for i in range(X_old.shape[0]):

		def ackley(solution):
			x = solution.get_x()
			query = (f'Feature 0: {x[0]}\n' +
			f'Feature 1: {x[1]}\n' +
			f'Output: ')

			text = instruction + prompt + query

			response = client.chat.completions.create(
				model=MODEL_NAME,
				messages=[
					{"role": "user", "content": text}
				],
				temperature=0,
				extra_body = {'do_sample':False}
			)
			predicted = response.choices[0].message.content
			if predicted == '1':
				predicted = 1.0
			if predicted == '0':
				predicted = 0.0
			x_old = np.array([X_old[i][0], X_old[i][1]]).astype(float)
			x_new = np.array([x[0], x[1]]).astype(float)
			cost = x_new - x_old
			loss = np.linalg.norm(np.array([predicted-target]),1) + lamb * np.linalg.norm(cost,1)

			return loss

		dim_list = [
			(ValueType.CONTINUOUS,  [-3, 3], 1e-2),
			(ValueType.CONTINUOUS,  [-3, 3], 1e-2),
		]

		dim = Dimension2(dim_list)

		objective = Objective(ackley, dim)
		# form up the objective function
		parameter = Parameter(budget=1000)

		sol = Opt.min(objective, parameter)
		answer = sol.get_x()
		predicted_recourse.append(answer)
		loss.append(sol.get_value())
	return np.array(predicted_recourse), np.array(loss)



def SimulatedData_choose_lambda(X_old, target,prompt, MODEL_NAME, API_KEY, BASE_URL):
	lambdas = np.arange(0.1, 1.1, 0.1)
	v_old = 0
	recourse_history = []
	validity_history = []
	loss = []
	for i, lamb in tqdm(enumerate(lambdas)):
		print('测试lamda中')
		print("Testing lambda:%f" % lamb)
		r,l = SimulatedData_recourse(X_old,target=target, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL, lamb=lamb)
		v = SimulatedData_ICL_recourse_validity(X_recourse=r, target=target, prompt=prompt, MODEL_NAME=MODEL_NAME, API_KEY=API_KEY, BASE_URL=BASE_URL)
		recourse_history.append(r)
		validity_history.append(v)
		loss.append(l)
		print(f'lamb:{lamb}, validity:{v}, loss:{l}')
		if v > v_old:
			v_old = v
		else:
			li = max(0, i - 1)
			print(f'search end,the finding is {lambdas[li]}')
			return lambdas[li], recourse_history[li], validity_history[li], np.mean(loss[li])
	print(f'search end,the finding is {lamb}')
	return lamb, recourse_history[-1], validity_history[-1], np.mean(loss[-1])




if __name__ == "__main__":

	print('')