class BlindingConductor(Conductor):

  def parseData(args):

    studyId, standardKeys, jobId = args['studyId'], args['standardKeys'], args['jobId']

    log.debug('sending Started event')

    parentEventId = await this.blindingEventSender.sendEvent({
      'studyId': studyId,
      'batchJobId': jobId,
      'parentEventId': None,
      'description': 'IQ Blinding process started',
      'state': 'Committed',
      'blindingData': BlindingArea.Issues
    })

    log.debug('sent Started event')
    try:
      await this.applyIssuesBlinding(studyId, standardKeys)

      log.debug('sending Successful event')

      await this.blindingEventSender.sendEvent({
        'studyId': studyId,
        'batchJobId': jobId,
        'parentEventId': parentEventId,
        'description': 'IQ Blinding process completed successfully',
        'state': 'Successful',
        'blindingData': BlindingArea.Issues
      })

      log.debug('sent Successful event')
    except Exception as e:
      log.debug('sending Failed event')

      await this.blindingEventSender.sendEvent({
        'studyId': studyId,
        'batchJobId': jobId,
        'parentEventId': parentEventId,
        'description': 'IQ Blinding process failed',
        'state': 'Failed',
        'blindingData': BlindingArea.Issues
      })

      log.debug('sent Failed event')

      raise e
