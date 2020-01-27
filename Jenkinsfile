pipeline {
  agent { label 'staging' }

  stages {
    stage('pylint') {
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
