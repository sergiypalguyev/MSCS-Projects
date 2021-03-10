from app import app
from flask import render_template, jsonify, request
import fhirbase
import psycopg2
from flask_json import FlaskJSON, JsonError, json_response, as_json
from collections import namedtuple
import json
from CodingObj import CodingObj
books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'},
    {'id': 1,
     'title': 'The Ones Who Walk Away From Omelas',
     'author': 'Ursula K. Le Guin',
     'first_sentence': 'With a clamor of bells that set the swallows soaring, the Festival of Summer came to the city Omelas, bright-towered by the sea.',
     'published': '1973'},
    {'id': 2,
     'title': 'Dhalgren',
     'author': 'Samuel R. Delany',
     'first_sentence': 'to wound the autumnal city.',
     'published': '1975'}
]

determineTestStr = CodingObj("http://www.ama-assn.org/go/cpt", "7D2648", "DETERMINE™ HIV-1/2 AG/AB COMBO")
uniGoldTestStr = CodingObj("http://www.ama-assn.org/go/cpt", "86701QW", "Uni-Gold Recombigen HIV")
sdBiolineTestStr = CodingObj("http://www.ama-assn.org/go/cpt", "02FK16", "Uni-Gold Recombigen HIV")

def testFormDecoder(formDict):
    return namedtuple('X', formDict.keys())(*formDict.values())

@app.route('/test')
def home():
    return "<b>There has been a change</b>"

@app.route('/template')
def template():
    return render_template('home.html')

@app.route('/api/v1/resources/books', methods=['GET']) 
def api_poc():
    return jsonify(books)


@app.route('/api/v1/resources/allPatient', methods=['GET']) 
def get_resource():
    connection = psycopg2.connect(
    database='fhirbase', user='postgress',
    host='fhirbasedb-service.globalhivteam30.svc.cluster.local', port='5432', password='postgres')
    fb = fhirbase.FHIRBase(connection)
    #patientData = fb.read(resource_type='Patient', id='patientId')
    with fb.execute('SELECT * FROM patient') as cursor: 
        respons = (cursor.fetchall())
        return jsonify(respons)

@app.route('/api/v1/resources/testForm', methods=['POST']) 
def POST_resource():
    #have new test
    connection = psycopg2.connect(
    database='fhirbase', user='postgress',
    host='172.17.0.3', port='5432', password='postgres')
    fb = fhirbase.FHIRBase(connection)
    input_form = request.get_json()
    #transform
    #form_obj = json.loads(input_form, object_hook=testFormDecoder)
    #return some sample values for display
    testForm = input_form['testForm']
    print('\nfirst name : ', testForm['name']['firstName'])
    fb = fhirbase.FHIRBase(connection)
    resultOfPatientCreation = fb.create({
        'resourceType': 'Patient',
        'name':[{'use':'official', 'family':testForm['name']['lastName'], 'given': [testForm['name']['firstName'], testForm['name']['middleName']]}],
        'gender': testForm['sex'],
        'birthDate': testForm['dob'],

    })
    print('\nresult of create : ', resultOfPatientCreation)
    mapOfTests = {}
    for i in testForm['testData']['tests']:
        print("\n loping over map:  ", i)
        if i['testNumber'] == 1:
            mapOfTests[0] = i
        elif i['testNumber'] == 2:
            mapOfTests[1] = i
        elif i['testNumber'] == 3:
            mapOfTests[2] = i

    print("\nwhat is in the map of tests: ", len(mapOfTests))

    firstTest = returnTestTypeData(testForm['testData']['tests'][0]['testAssay'])
    secondTest = returnTestTypeData(testForm['testData']['tests'][1]['testAssay'])
    thirdTest = returnTestTypeData(testForm['testData']['tests'][2]['testAssay'])

    idOfPatient = resultOfPatientCreation['id']
    
    print("\nID of patient: ", idOfPatient)

    resultOfHIVObservationCreation = fb.create({
        'resourceType': 'Observation',
        'code':{'coding':[{'system': 'http://loinc.org', 'code': '69668-2', 'display': 'HIV 1 and 2 Ab IA.rapid Nom'}]},
        'subject': {'reference': 'Patient/' + str(idOfPatient)},
        'effectiveDateTime': testForm['testData']['testDate'],
        'valueString': testForm['resultReceived'],
        'component': [{'code': {'coding':[{'system': firstTest.system, 'code':  firstTest.code, 'display':  firstTest.display}]}, 'valueString': mapOfTests[0]['testResult']},
        {'code': {'coding':[{'system': secondTest.system, 'code':  secondTest.code, 'display':  secondTest.display}]}, 'valueString': mapOfTests[1]['testResult']},
        {'code': {'coding':[{'system': thirdTest.system, 'code':  thirdTest.code, 'display':  thirdTest.display}]}, 'valueString': mapOfTests[2]['testResult']}]
    })
    print('\nresult of create : ', resultOfHIVObservationCreation)
    sampleAns = str(resultOfPatientCreation) + "\n" + str(resultOfHIVObservationCreation)
    return sampleAns
    #connection = psycopg2.connect(
    #database='fhirbase', user='postgress',
    #host='fhirbasedb-service.globalhivteam30.svc.cluster.local', port='5432', password='postgres')
    #fb = fhirbase.FHIRBase(connection)
    #patientData = fb.read(resource_type='Patient', id='patientId')
    #with fb.execute('SELECT * FROM patient') as cursor: 
        #respons = (cursor.fetchall())
        #return jsonify(respons)

def returnTestTypeData(testName: str):
    if testName.lower() == 'determine':
        return determineTestStr
    elif testName.lower() == 'uni-gold recombigen':
        return uniGoldTestStr
    elif testName.lower() == 'sd bioline':
        return sdBiolineTestStr
    else: 
        return None

@app.route('/api/v1/resources/patient/<id>', methods=['GET']) 
def get_patient_resource(id):
    connection = psycopg2.connect(
    database='fhirbase', user='postgress',
    host='172.17.0.3', port='5432', password='postgres')
    fb = fhirbase.FHIRBase(connection)
    #patientData = fb.read(resource_type='Patient', id='patientId')
    #with fb.execute('SELECT * FROM patient WHERE id=%s', [id]) as cursor: 
        #return(cursor.fetchall())
    print('\nID of Patient:  ', [id])
    getPatient = fb.read({
        'resourceType': 'Patient',
        'id': [id]
    })
    print('\nGot Patient:  ', getPatient)




