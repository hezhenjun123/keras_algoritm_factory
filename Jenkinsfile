pipeline {
  agent { label 'staging' }

  stages {
  stage('pytest') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh python -m pytest
        '''
      }
    }
    stage('trt_yield') {
      steps {
        sh '''#!/bin/bash
        bash run_docker.sh bash regression_test.sh
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
