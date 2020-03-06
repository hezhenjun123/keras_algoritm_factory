pipeline {
  agent { label 'zl' }

  stages {
  stage('pytest') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh python -m pytest && \
        cd ..
        '''
      }
    }
    stage('yield') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 1 && \
        cd ..
        '''
      }
    }
    stage('chaff hopper') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 2 && \
        cd ..
        '''
      }
    }

    stage('chaff lodging') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 3 && \
        cd ..
        '''
      }
    }

    stage('chaff elevator trt') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 4 && \
        cd ..
        '''
      }
    }

    stage('trt load test') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 5 && \
        cd ..

        '''
      }
    }
    stage('trt load test fp_16') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 6 && \
        cd ..
        '''
      }
    }
    stage('TF lodging model') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 7 && \
        cd ..
        '''
      }
    }
    stage(' TF chaff hopper model') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 8 && \
        cd ..
        '''
      }
    }
    stage('TF chaff elevator model') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 9 && \
        cd ..
        '''
      }
    }
    stage('TF yield model') {
      steps {
        sh '''#!/bin/bash
        cd scripts && \
        bash run_docker.sh bash regression_test.sh -s 10 && \
        cd ..
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
