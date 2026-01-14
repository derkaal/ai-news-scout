# Incident Response Runbook

## Newsletter Clustering Agent - Production Incident Response

This runbook provides step-by-step procedures for responding to production incidents in the newsletter clustering system.

---

## 🚨 Emergency Response Overview

### Incident Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P0 - Critical** | System down, data loss, security breach | 15 minutes | Immediate |
| **P1 - High** | Major functionality impaired, SLA breach | 1 hour | Within 2 hours |
| **P2 - Medium** | Minor functionality impaired, degraded performance | 4 hours | Next business day |
| **P3 - Low** | Cosmetic issues, minor bugs | 24 hours | Weekly review |

### Emergency Contacts

- **On-Call Engineer**: [Phone] [Email]
- **Backup Engineer**: [Phone] [Email]
- **System Administrator**: [Phone] [Email]
- **Product Owner**: [Phone] [Email]

---

## 📋 General Incident Response Process

### 1. Incident Detection
- **Automated Alerts**: Monitor alert channels (email, Slack, dashboard)
- **Manual Detection**: User reports, monitoring dashboard observations
- **Health Check Failures**: Load balancer or orchestrator alerts

### 2. Initial Response (Within 5 minutes)
1. **Acknowledge the incident** in monitoring system
2. **Assess severity** using the criteria below
3. **Create incident ticket** with initial details
4. **Notify stakeholders** based on severity level
5. **Begin investigation** using this runbook

### 3. Investigation and Resolution
1. **Gather information** using monitoring tools
2. **Identify root cause** using diagnostic procedures
3. **Implement fix** following change procedures
4. **Verify resolution** using health checks
5. **Monitor for recurrence** for 30 minutes

### 4. Post-Incident
1. **Update stakeholders** on resolution
2. **Document incident** in incident log
3. **Schedule post-mortem** (P0/P1 incidents)
4. **Implement preventive measures** if needed

---

## 🔍 Diagnostic Procedures

### Quick Health Check Commands

```bash
# Check overall system status
curl http://localhost:8080/health

# Check detailed health information
curl http://localhost:8080/health/detailed

# Check metrics summary
curl http://localhost:8080/metrics/summary

# Check performance status
curl http://localhost:8080/performance

# Check quality metrics
curl http://localhost:8080/quality
```

### Log Analysis Commands

```bash
# Check recent application logs
tail -f ~/.newsletter_agent/logs/monitoring.log

# Search for errors in logs
grep -i "error\|exception\|failed" ~/.newsletter_agent/logs/monitoring.log | tail -20

# Check system resource usage
top -p $(pgrep -f newsletter_agent)
free -h
df -h
```

---

## 🚨 Specific Incident Scenarios

### Scenario 1: System Completely Down

**Symptoms:**
- Health check endpoints not responding
- Application not processing newsletters
- All monitoring alerts firing

**Immediate Actions:**
1. **Check if process is running:**
   ```bash
   ps aux | grep newsletter_agent
   systemctl status newsletter-agent  # if using systemd
   ```

2. **Check system resources:**
   ```bash
   free -h
   df -h
   top
   ```

3. **Check application logs:**
   ```bash
   tail -50 ~/.newsletter_agent/logs/monitoring.log
   ```

4. **Restart application if needed:**
   ```bash
   # Stop gracefully first
   pkill -TERM -f newsletter_agent
   
   # Wait 30 seconds, then force kill if needed
   pkill -KILL -f newsletter_agent
   
   # Restart application
   python -m newsletter_agent_core.agent
   ```

5. **Verify recovery:**
   ```bash
   curl http://localhost:8080/health
   ```

**Escalation:** If restart doesn't resolve, escalate to P0 and contact system administrator.

---

### Scenario 2: High Memory Usage

**Symptoms:**
- Memory usage > 2GB sustained
- Memory limit exceeded alerts
- System becoming unresponsive

**Immediate Actions:**
1. **Check current memory usage:**
   ```bash
   curl http://localhost:8080/performance | jq '.memory'
   free -h
   ```

2. **Identify memory-consuming processes:**
   ```bash
   ps aux --sort=-%mem | head -10
   ```

3. **Check for memory leaks:**
   ```bash
   # Monitor memory over time
   watch -n 5 'curl -s http://localhost:8080/performance | jq ".memory.current_mb"'
   ```

4. **Immediate mitigation:**
   ```bash
   # Clear application caches
   curl -X POST http://localhost:8080/admin/clear-cache
   
   # Or restart if critical
   systemctl restart newsletter-agent
   ```

5. **Monitor recovery:**
   - Watch memory usage for 15 minutes
   - Ensure processing continues normally
   - Check for recurring memory growth

**Root Cause Investigation:**
- Check for large datasets being processed
- Review recent code changes
- Analyze memory usage patterns in monitoring

---

### Scenario 3: Processing Time Exceeded

**Symptoms:**
- Processing time > 30 seconds consistently
- Timeout alerts firing
- Clustering operations failing

**Immediate Actions:**
1. **Check current processing performance:**
   ```bash
   curl http://localhost:8080/performance | jq '.operations'
   ```

2. **Check system load:**
   ```bash
   uptime
   iostat 1 5
   ```

3. **Review recent processing:**
   ```bash
   curl http://localhost:8080/quality | jq '.current_metrics'
   ```

4. **Immediate mitigation:**
   ```bash
   # Reduce batch size temporarily
   export CLUSTERING_CHUNK_SIZE=50
   
   # Switch to faster algorithm if needed
   export CLUSTERING_DEFAULT_ALGORITHM=hdbscan
   
   # Restart with new settings
   systemctl restart newsletter-agent
   ```

5. **Monitor improvement:**
   - Check processing times for next few operations
   - Verify quality metrics remain acceptable

**Root Cause Investigation:**
- Check input data size and complexity
- Review algorithm performance
- Analyze system resource constraints

---

### Scenario 4: Poor Clustering Quality

**Symptoms:**
- Silhouette score < 0.2
- Noise ratio > 30%
- Quality alerts firing

**Immediate Actions:**
1. **Check current quality metrics:**
   ```bash
   curl http://localhost:8080/quality
   ```

2. **Review recent clustering results:**
   ```bash
   curl http://localhost:8080/quality?trends=true&trends_window=24
   ```

3. **Check input data quality:**
   - Review recent newsletter content
   - Check for data anomalies
   - Verify embedding generation

4. **Immediate mitigation:**
   ```bash
   # Switch to more conservative algorithm
   export CLUSTERING_DEFAULT_ALGORITHM=hierarchical
   
   # Adjust clustering parameters
   export CLUSTERING_MIN_CLUSTER_SIZE=5
   
   # Restart with new settings
   systemctl restart newsletter-agent
   ```

5. **Validate improvement:**
   - Process test batch
   - Check quality metrics
   - Verify results make sense

**Root Cause Investigation:**
- Analyze input data characteristics
- Review algorithm parameter tuning
- Check for embedding model issues

---

### Scenario 5: Google API Integration Failure

**Symptoms:**
- Gmail/Sheets API errors
- Authentication failures
- Data not being written to sheets

**Immediate Actions:**
1. **Check API connectivity:**
   ```bash
   # Test Gmail API
   python -c "from newsletter_agent_core.agent import get_gmail_service; print(get_gmail_service())"
   
   # Test Sheets API
   python -c "from newsletter_agent_core.agent import get_sheets_service; print(get_sheets_service())"
   ```

2. **Check credentials:**
   ```bash
   ls -la credentials.json token.json
   cat .env | grep GOOGLE
   ```

3. **Check API quotas:**
   - Review Google Cloud Console
   - Check for quota exceeded errors
   - Verify billing status

4. **Immediate mitigation:**
   ```bash
   # Refresh OAuth token
   rm token.json
   # Restart application to re-authenticate
   systemctl restart newsletter-agent
   ```

5. **Verify recovery:**
   - Test newsletter processing
   - Check Google Sheets for new data
   - Monitor for recurring issues

**Root Cause Investigation:**
- Review API usage patterns
- Check for quota limits
- Verify credential configuration

---

### Scenario 6: Cache Performance Issues

**Symptoms:**
- Cache hit rate < 70%
- Slow embedding generation
- High processing times

**Immediate Actions:**
1. **Check cache statistics:**
   ```bash
   curl http://localhost:8080/metrics/summary | jq '.cache_stats'
   ```

2. **Check cache storage:**
   ```bash
   du -sh ~/.newsletter_agent/embeddings/
   ls -la ~/.newsletter_agent/embeddings/
   ```

3. **Immediate mitigation:**
   ```bash
   # Clear and rebuild cache if corrupted
   rm -rf ~/.newsletter_agent/embeddings/*
   
   # Or increase cache size
   export CLUSTERING_CACHE_SIZE_MB=1000
   
   # Restart application
   systemctl restart newsletter-agent
   ```

4. **Monitor improvement:**
   - Watch cache hit rate over next hour
   - Check processing time improvements

**Root Cause Investigation:**
- Analyze cache usage patterns
- Check for cache corruption
- Review cache configuration

---

## 🔧 Recovery Procedures

### Graceful Restart Procedure

```bash
# 1. Send graceful shutdown signal
pkill -TERM -f newsletter_agent

# 2. Wait for graceful shutdown (up to 30 seconds)
sleep 30

# 3. Force kill if still running
pkill -KILL -f newsletter_agent

# 4. Clear any locks or temporary files
rm -f /tmp/newsletter_agent.lock

# 5. Start application
python -m newsletter_agent_core.agent &

# 6. Wait for startup
sleep 10

# 7. Verify health
curl http://localhost:8080/health
```

### Emergency Rollback Procedure

```bash
# 1. Stop current version
systemctl stop newsletter-agent

# 2. Backup current configuration
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# 3. Restore previous version
git checkout HEAD~1  # or specific commit
pip install -r requirements_clustering.txt

# 4. Restore previous configuration
cp .env.previous .env

# 5. Start previous version
systemctl start newsletter-agent

# 6. Verify rollback success
curl http://localhost:8080/health
```

### Data Recovery Procedures

#### Recover from Google Sheets Backup
```bash
# 1. Export current sheet data
python -c "
from newsletter_agent_core.agent import get_sheets_service
service = get_sheets_service()
# Export logic here
"

# 2. Restore from backup
# Manual process through Google Sheets interface
```

#### Recover Cache Data
```bash
# 1. Backup current cache
cp -r ~/.newsletter_agent/embeddings ~/.newsletter_agent/embeddings.backup

# 2. Restore from backup
cp -r ~/.newsletter_agent/embeddings.backup.YYYYMMDD ~/.newsletter_agent/embeddings

# 3. Restart application
systemctl restart newsletter-agent
```

---

## 📊 Monitoring and Alerting

### Key Metrics to Monitor During Incidents

1. **System Health**
   - Overall status (healthy/degraded/unhealthy)
   - Uptime percentage
   - Error rate

2. **Performance Metrics**
   - Processing time (P50, P95, P99)
   - Memory usage (current, peak)
   - CPU utilization

3. **Quality Metrics**
   - Silhouette score
   - Noise ratio
   - Cluster coherence

4. **Business Metrics**
   - Success rate
   - Cache hit rate
   - Throughput

### Alert Acknowledgment

```bash
# Acknowledge alert via API (if implemented)
curl -X POST http://localhost:8080/alerts/acknowledge \
  -H "Content-Type: application/json" \
  -d '{"alert_id": "ALERT_ID", "acknowledged_by": "engineer_name"}'
```

---

## 📝 Incident Documentation Template

### Incident Report Template

```
Incident ID: INC-YYYY-NNNN
Date: YYYY-MM-DD
Time: HH:MM UTC
Severity: P0/P1/P2/P3
Status: Open/Investigating/Resolved/Closed

## Summary
Brief description of the incident

## Timeline
- HH:MM - Incident detected
- HH:MM - Initial response started
- HH:MM - Root cause identified
- HH:MM - Fix implemented
- HH:MM - Resolution verified

## Impact
- Services affected:
- Users affected:
- Duration:
- Data loss: Yes/No

## Root Cause
Detailed explanation of what caused the incident

## Resolution
Steps taken to resolve the incident

## Prevention
Measures to prevent recurrence

## Lessons Learned
What we learned from this incident

## Action Items
- [ ] Action item 1 (Owner: Name, Due: Date)
- [ ] Action item 2 (Owner: Name, Due: Date)
```

---

## 🔄 Post-Incident Procedures

### Post-Mortem Process (P0/P1 Incidents)

1. **Schedule post-mortem meeting** within 24 hours
2. **Gather all stakeholders** (engineering, product, operations)
3. **Review incident timeline** and response
4. **Identify root cause** and contributing factors
5. **Define action items** to prevent recurrence
6. **Update procedures** and documentation
7. **Share learnings** with broader team

### Incident Metrics to Track

- **MTTR** (Mean Time To Recovery)
- **MTTD** (Mean Time To Detection)
- **Incident frequency** by category
- **SLA compliance** during incidents
- **Alert accuracy** (false positive rate)

### Continuous Improvement

1. **Monthly incident review** meetings
2. **Quarterly runbook updates**
3. **Annual disaster recovery drills**
4. **Regular training** on incident response
5. **Tool and process improvements**

---

## 📞 Escalation Matrix

| Incident Type | Primary Contact | Secondary Contact | Management |
|---------------|----------------|-------------------|------------|
| System Down | On-Call Engineer | Backup Engineer | Engineering Manager |
| Performance Issues | On-Call Engineer | System Admin | Product Owner |
| Data Issues | On-Call Engineer | Data Engineer | Engineering Manager |
| Security Issues | Security Team | On-Call Engineer | CISO |
| API Issues | On-Call Engineer | Integration Team | Product Owner |

---

**Last Updated**: [Date]
**Version**: 1.0
**Next Review**: [Date + 3 months]