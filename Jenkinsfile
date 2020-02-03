pipeline {
  agent { label 'zl' }

  stages {
  stage('pytest') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh python -m pytest
        '''
      }
    }
    stage('yield') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh bash regression_test.sh -s 1
        '''
      }
    }
    stage('chaff hopper') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh bash regression_test.sh -s 2
        '''
      }
    }

    stage('chaff lodging') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh bash regression_test.sh -s 3
        '''
      }
    }

    stage('chaff elevator') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh bash regression_test.sh -s 4
        '''
      }
    }

    stage('trt load test') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh bash regression_test.sh -s 5
        '''
      }
    }
  }
  post {
    always {
      sh 'rm -rf *'
    }
  }
}
