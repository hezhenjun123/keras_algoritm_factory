pipeline {
  agent { label 'zl' }

  stages {
  stage('pytest') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh python -m pytest
        '''
      }
    }
    stage('yield') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 1
        '''
      }
    }
    stage('chaff hopper') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 2
        '''
      }
    }

    stage('chaff lodging') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 3
        '''
      }
    }

    stage('chaff elevator') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 4
        '''
      }
    }

    stage('trt load test') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 5
        '''
      }
    }
    stage('trt load test fp_16') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 6
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
