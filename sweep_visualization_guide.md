# W&B Sweep Visualization Guide for Knockout Diversity

## Data Structure Analysis

Your sweep collects the following metrics for each `knockout_diversity` value:

### Primary Metrics (Logged Every `periodic_eval_interval` epochs)
- `eval_ko_in/final_loss` - IN-distribution knockout loss
- `eval_ko_in/final_hard_loss` - IN-distribution knockout hard loss  
- `eval_ko_in/final_accuracy` - IN-distribution knockout accuracy
- `eval_ko_in/final_hard_accuracy` - IN-distribution knockout hard accuracy
- `eval_ko_out/final_loss` - OUT-of-distribution knockout loss
- `eval_ko_out/final_hard_loss` - OUT-of-distribution knockout hard loss
- `eval_ko_out/final_accuracy` - OUT-of-distribution knockout accuracy
- `eval_ko_out/final_hard_accuracy` - OUT-of-distribution knockout hard accuracy

### Debug Metrics
- `debug/in_knockout_patterns_sum` - Sum of IN-distribution patterns
- `debug/out_knockout_patterns_sum` - Sum of OUT-of-distribution patterns
- `debug/patterns_are_identical` - Boolean flag for pattern identity
- `debug/id_patterns_in_vocab_count` - Count of ID patterns in vocabulary
- `debug/ood_patterns_in_vocab_count` - Count of OOD patterns in vocabulary

## Recommended W&B Visualizations

### 1. Line Plot: Knockout Performance vs Training Progress
**Chart Type**: Line Plot  
**X-axis**: `eval_ko_in/epoch`  
**Y-axis**: `eval_ko_in/final_accuracy`  
**Group By**: `pool.persistent_knockout.knockout_diversity`  
**Additional Lines**: `eval_ko_out/final_accuracy`

**What it shows**: How different vocabulary sizes affect recovery performance over training time.

### 2. Scatter Plot: IN vs OUT Distribution Performance
**Chart Type**: Scatter Plot  
**X-axis**: `eval_ko_in/final_accuracy`  
**Y-axis**: `eval_ko_out/final_accuracy`  
**Color**: `pool.persistent_knockout.knockout_diversity`  
**Size**: `eval_ko_in/epoch` (optional)

**What it shows**: Generalization gap between IN-distribution and OUT-of-distribution patterns for each diversity level.

### 3. Bar Chart: Final Performance Comparison
**Chart Type**: Bar Chart  
**X-axis**: `pool.persistent_knockout.knockout_diversity`  
**Y-axis**: `eval_ko_in/final_accuracy` (or loss)  
**Group By**: `eval_ko_in/epoch` (use final epoch values)

**What it shows**: Direct comparison of final performance across different vocabulary sizes.

### 4. Heatmap: Performance Matrix
**Chart Type**: Heatmap  
**X-axis**: `pool.persistent_knockout.knockout_diversity`  
**Y-axis**: `eval_ko_in/epoch` (binned)  
**Color**: `eval_ko_in/final_accuracy`

**What it shows**: Performance evolution across both diversity and training time dimensions.

### 5. Custom Panel: Recovery Trajectory Analysis
**Chart Type**: Custom Panel (if available)  
**Metrics**: 
- `eval_ko_in/final_accuracy` vs `eval_ko_out/final_accuracy` ratio
- `eval_ko_in/final_loss` vs `eval_ko_out/final_loss` ratio

**What it shows**: How well models generalize beyond their training vocabulary.

## Advanced Analysis Questions

### 1. Vocabulary Size vs Generalization
- **Question**: Does larger vocabulary size improve OUT-of-distribution performance?
- **Visualization**: Plot `eval_ko_out/final_accuracy` vs `pool.persistent_knockout.knockout_diversity`
- **Expected Pattern**: Larger vocabularies should show better OOD performance

### 2. Training Stability Analysis
- **Question**: Does vocabulary size affect training stability?
- **Visualization**: Plot standard deviation of `eval_ko_in/final_accuracy` across epochs vs `pool.persistent_knockout.knockout_diversity`
- **Expected Pattern**: Larger vocabularies might show more stable training

### 3. Convergence Speed Analysis
- **Question**: How quickly do different vocabulary sizes converge?
- **Visualization**: Plot epoch when `eval_ko_in/final_accuracy` first exceeds threshold vs `pool.persistent_knockout.knockout_diversity`
- **Expected Pattern**: Larger vocabularies might converge slower but to better final performance

### 4. Overfitting Detection
- **Question**: Do larger vocabularies prevent overfitting to specific patterns?
- **Visualization**: Plot gap between `eval_ko_in/final_accuracy` and `eval_ko_out/final_accuracy` vs `pool.persistent_knockout.knockout_diversity`
- **Expected Pattern**: Smaller gaps indicate better generalization

## W&B Query Examples

### Query 1: Final Performance Comparison
```sql
SELECT 
  config.pool.persistent_knockout.knockout_diversity,
  AVG(eval_ko_in.final_accuracy) as avg_in_accuracy,
  AVG(eval_ko_out.final_accuracy) as avg_out_accuracy,
  STDDEV(eval_ko_in.final_accuracy) as std_in_accuracy
FROM runs 
WHERE config.pool.persistent_knockout.knockout_diversity IS NOT NULL
GROUP BY config.pool.persistent_knockout.knockout_diversity
ORDER BY config.pool.persistent_knockout.knockout_diversity
```

### Query 2: Training Trajectory Analysis
```sql
SELECT 
  eval_ko_in.epoch,
  config.pool.persistent_knockout.knockout_diversity,
  AVG(eval_ko_in.final_accuracy) as avg_in_accuracy,
  AVG(eval_ko_out.final_accuracy) as avg_out_accuracy
FROM runs 
WHERE config.pool.persistent_knockout.knockout_diversity IS NOT NULL
  AND eval_ko_in.epoch IS NOT NULL
GROUP BY eval_ko_in.epoch, config.pool.persistent_knockout.knockout_diversity
ORDER BY eval_ko_in.epoch, config.pool.persistent_knockout.knockout_diversity
```

### Query 3: Generalization Gap Analysis
```sql
SELECT 
  config.pool.persistent_knockout.knockout_diversity,
  AVG(eval_ko_in.final_accuracy - eval_ko_out.final_accuracy) as generalization_gap,
  AVG(eval_ko_in.final_loss - eval_ko_out.final_loss) as loss_gap
FROM runs 
WHERE config.pool.persistent_knockout.knockout_diversity IS NOT NULL
GROUP BY config.pool.persistent_knockout.knockout_diversity
ORDER BY config.pool.persistent_knockout.knockout_diversity
```

## Expected Insights

### Hypothesis 1: Vocabulary Size vs Performance
- **Small vocabularies (1, 16)**: May show good IN-distribution performance but poor generalization
- **Medium vocabularies (128, 512)**: Balanced performance with reasonable generalization
- **Large vocabularies (1024)**: Best generalization but potentially slower convergence

### Hypothesis 2: Training Dynamics
- **Small vocabularies**: Faster initial convergence, potential overfitting
- **Large vocabularies**: Slower convergence, better final generalization
- **Medium vocabularies**: Optimal balance of convergence speed and final performance

### Hypothesis 3: Pattern Diversity Effects
- **Vocabulary size 1**: Essentially no diversity, should show poor generalization
- **Vocabulary size 16**: Limited diversity, may show some generalization
- **Vocabulary size 128+**: Sufficient diversity for good generalization

## Implementation Notes

1. **Data Collection**: Your current setup logs metrics every `periodic_eval_interval` epochs (1024), which provides good temporal resolution
2. **Batch Size**: Each evaluation uses `periodic_eval_batch_size` circuits (should match `knockout_diversity`)
3. **Pattern Sampling**: IN-distribution patterns are sampled from vocabulary, OUT-distribution patterns are generated fresh
4. **Debug Metrics**: Use debug metrics to verify pattern sampling is working correctly

This visualization strategy will help you understand how knockout pattern vocabulary size affects both training dynamics and generalization performance in your Boolean NCA system. 