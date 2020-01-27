pipeline {
  agent { label 'staging' }

  stages {
  stage('pytest') {
      steps {
        sh '''#!/bin/bash
        pip install pytest && pytest
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
