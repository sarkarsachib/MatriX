import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import asyncio
import aiohttp
import schedule
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path
import threading
import queue
import pickle

# Import our components
from neural_core.advanced_neural_core import MaxedOutSathikNeuralCore
from web_crawler.web_crawler_unit import BasicSpider
from web_crawler.raw_data_processor import RawDataProcessor
from web_crawler.tokenizer import BPETokenizer
from memory_system.memory_system import ShortTermMemory, LongTermMemory, SelfHealingLayer
from memory_system.safety_modules import TruthComparator, ContentFilter, Obfuscator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebDataset(Dataset):
    """Dataset for web-crawled data"""
    def __init__(self, data_path: str, tokenizer: BPETokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str):
        """Load processed web data"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Data file {data_path} not found. Using empty dataset.")
            return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        content = item.get('content', '')
        
        # Tokenize content
        tokens = self.tokenizer.encode(content)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([0] * (self.max_length - len(tokens)))  # Pad with 0
        
        # Create input and target (shifted by 1 for language modeling)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'metadata': item.get('metadata', {})
        }

class LiveTrainingLoop:
    """Advanced live training loop for Sathik AI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_components()
        
        # Training state
        self.training_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Data queues for real-time processing
        self.data_queue = queue.Queue(maxsize=1000)
        self.feedback_queue = queue.Queue(maxsize=100)
        
        # Training metrics
        self.metrics = {
            'loss_history': [],
            'learning_rate_history': [],
            'web_update_count': 0,
            'self_correction_count': 0
        }
        
        # Scheduler for periodic tasks
        self._setup_scheduler()
        
    def _initialize_components(self):
        """Initialize all AI components"""
        logger.info("Initializing Sathik AI components...")
        
        # Initialize tokenizer
        self.tokenizer = BPETokenizer(vocab_size=self.config.get('vocab_size', 50000))
        
        # Initialize neural core
        self.model = MaxedOutSathikNeuralCore(
            vocab_size=self.config.get('vocab_size', 50000),
            d_model=self.config.get('d_model', 2048),
            num_heads=self.config.get('num_heads', 32),
            num_layers=self.config.get('num_layers', 48),
            num_experts=self.config.get('num_experts', 128),
            top_k=self.config.get('top_k', 16)
        ).to(self.device)
        
        # Initialize optimizer with advanced scheduling
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('scheduler_t0', 1000),
            T_mult=2,
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        # Initialize data processor
        self.data_processor = RawDataProcessor()
        
        # Initialize memory systems
        self.short_term_memory = ShortTermMemory(max_size=1000)
        self.long_term_memory = LongTermMemory("sathik_ltm.json")
        self.self_healing = SelfHealingLayer(self.long_term_memory)
        
        # Initialize safety modules
        self.truth_comparator = TruthComparator()
        self.content_filter = ContentFilter()
        self.obfuscator = Obfuscator()
        
        # Set up reliable sources for truth comparison
        self._setup_truth_sources()
        
        logger.info("All components initialized successfully!")
    
    def _setup_truth_sources(self):
        """Setup reliability scores for different sources"""
        reliable_sources = {
            'wikipedia.org': 0.9,
            'arxiv.org': 0.95,
            'nature.com': 0.95,
            'science.org': 0.95,
            'ieee.org': 0.9,
            'acm.org': 0.9,
            'stackoverflow.com': 0.8,
            'github.com': 0.8,
            'reddit.com': 0.6,
            'twitter.com': 0.4,
            'facebook.com': 0.3
        }
        
        for source, reliability in reliable_sources.items():
            self.truth_comparator.add_source_reliability(source, reliability)
    
    def _setup_scheduler(self):
        """Setup periodic tasks"""
        # Web crawling every hour
        schedule.every().hour.do(self._crawl_web_data)
        
        # Model checkpoint every 6 hours
        schedule.every(6).hours.do(self._save_checkpoint)
        
        # Memory consolidation every day
        schedule.every().day.do(self._consolidate_memory)
        
        # Self-healing check every 2 hours
        schedule.every(2).hours.do(self._perform_self_healing)
        
        # Performance evaluation every 12 hours
        schedule.every(12).hours.do(self._evaluate_performance)
    
    async def _crawl_web_data(self):
        """Crawl web for new training data"""
        logger.info("Starting web crawling for new data...")
        
        try:
            # Simulate web crawling (in real implementation, use actual scrapers)
            new_data = await self._simulate_web_crawling()
            
            # Process raw data
            processed_data = self.data_processor.process_batch(new_data)
            
            # Filter content for safety
            safe_data = []
            for item in processed_data:
                content_analysis = self.content_filter.analyze_content(item['content'])
                if content_analysis['is_safe']:
                    safe_data.append(item)
                else:
                    logger.warning(f"Filtered unsafe content: {content_analysis['issues']}")
            
            # Add to training queue
            for item in safe_data:
                if not self.data_queue.full():
                    self.data_queue.put(item)
            
            self.metrics['web_update_count'] += len(safe_data)
            logger.info(f"Added {len(safe_data)} new safe data items to training queue")
            
        except Exception as e:
            logger.error(f"Error during web crawling: {e}")
    
    async def _simulate_web_crawling(self) -> List[Dict[str, Any]]:
        """Simulate web crawling (replace with actual implementation)"""
        # This is a simulation - in real implementation, use actual web scrapers
        sample_data = [
            {
                'content': f'Sample web content {i} about AI and machine learning.',
                'url': f'https://example.com/page{i}',
                'timestamp': datetime.now().isoformat(),
                'content_type': 'text'
            }
            for i in range(10)
        ]
        return sample_data
    
    def _perform_self_healing(self):
        """Perform self-healing and error correction"""
        logger.info("Performing self-healing checks...")
        
        try:
            # Check for inconsistencies in long-term memory
            concepts_to_check = list(self.long_term_memory.memory.keys())[:10]  # Check first 10
            
            corrections_made = 0
            for concept_id in concepts_to_check:
                concept_data = self.long_term_memory.get_concept(concept_id)
                if concept_data:
                    # Simulate new data for validation
                    new_data = {
                        'content': concept_data['data'].get('content', ''),
                        'timestamp': datetime.now().isoformat(),
                        'confidence': np.random.random()  # Simulate confidence score
                    }
                    
                    # Attempt correction
                    if self.self_healing.verify_and_correct(concept_id, new_data):
                        corrections_made += 1
            
            self.metrics['self_correction_count'] += corrections_made
            logger.info(f"Self-healing completed. Made {corrections_made} corrections.")
            
        except Exception as e:
            logger.error(f"Error during self-healing: {e}")
    
    def _consolidate_memory(self):
        """Consolidate short-term memory into long-term memory"""
        logger.info("Consolidating memory...")
        
        try:
            # Get recent short-term memories
            recent_memories = self.short_term_memory.get_recent_entries(50)
            
            # Process and store important memories in long-term memory
            for memory in recent_memories:
                content = memory.get('content', {})
                if isinstance(content, dict) and content.get('importance', 0) > 0.7:
                    # Store in long-term memory
                    concept_id = f"concept_{int(time.time())}_{hash(str(content)) % 10000}"
                    self.long_term_memory.add_concept(concept_id, content)
            
            # Clear processed short-term memories
            self.short_term_memory.clear()
            
            logger.info("Memory consolidation completed")
            
        except Exception as e:
            logger.error(f"Error during memory consolidation: {e}")
    
    def _save_checkpoint(self):
        """Save model checkpoint"""
        logger.info("Saving model checkpoint...")
        
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'training_step': self.training_step,
                'epoch': self.epoch,
                'best_loss': self.best_loss,
                'metrics': self.metrics,
                'config': self.config
            }
            
            checkpoint_path = f"checkpoints/sathik_checkpoint_{self.training_step}.pt"
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            
            # Save tokenizer
            self.tokenizer.save(f"checkpoints/sathik_tokenizer_{self.training_step}.pkl")
            
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _evaluate_performance(self):
        """Evaluate model performance"""
        logger.info("Evaluating model performance...")
        
        try:
            self.model.eval()
            
            # Generate sample text for evaluation
            sample_input = torch.randint(0, self.tokenizer.get_vocab_size(), (1, 10)).to(self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    sample_input,
                    max_length=50,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(generated[0].cpu().tolist())
                
                # Analyze generated text
                content_analysis = self.content_filter.analyze_content(generated_text)
                
                logger.info(f"Generated sample: {generated_text[:100]}...")
                logger.info(f"Content safety: {content_analysis['is_safe']}")
                logger.info(f"Current loss: {self.metrics['loss_history'][-1] if self.metrics['loss_history'] else 'N/A'}")
            
            self.model.train()
            
        except Exception as e:
            logger.error(f"Error during performance evaluation: {e}")
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        # Prepare memory vectors
        memory_vectors = {
            'short_term': torch.randn(input_ids.size(0), self.config.get('d_model', 2048)).to(self.device),
            'long_term': torch.randn(input_ids.size(0), self.config.get('d_model', 2048)).to(self.device)
        }
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            memory_vectors=memory_vectors,
            return_dict=True
        )
        
        # Calculate language modeling loss
        logits = outputs['outputs']['language_modeling']
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Add load balancing loss
        total_loss = loss + outputs['load_balancing_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update metrics
        self.metrics['loss_history'].append(loss.item())
        self.metrics['learning_rate_history'].append(self.scheduler.get_last_lr()[0])
        
        # Store training experience in short-term memory
        self.short_term_memory.add_entry({
            'type': 'training_step',
            'loss': loss.item(),
            'step': self.training_step,
            'importance': 1.0 - min(loss.item() / 10.0, 1.0)  # Higher importance for lower loss
        })
        
        self.training_step += 1
        
        return {
            'loss': loss.item(),
            'total_loss': total_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        epoch_losses = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Train step
            step_results = self.train_step(batch)
            epoch_losses.append(step_results['loss'])
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {self.epoch}, Step {batch_idx}: "
                    f"Loss = {step_results['loss']:.4f}, "
                    f"LR = {step_results['learning_rate']:.2e}"
                )
            
            # Process any new web data
            self._process_queued_data()
        
        avg_loss = np.mean(epoch_losses)
        
        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            logger.info(f"New best loss: {self.best_loss:.4f}")
        
        return avg_loss
    
    def _process_queued_data(self):
        """Process any new data from the web crawling queue"""
        processed_count = 0
        
        while not self.data_queue.empty() and processed_count < 10:  # Process up to 10 items
            try:
                new_item = self.data_queue.get_nowait()
                
                # Add to short-term memory for potential use
                self.short_term_memory.add_entry({
                    'type': 'web_data',
                    'content': new_item['content'],
                    'metadata': new_item['metadata'],
                    'importance': new_item.get('quality_score', 0.5)
                })
                
                processed_count += 1
                
            except queue.Empty:
                break
    
    def start_training(self, data_path: str, num_epochs: int = 1000):
        """Start the live training loop"""
        logger.info("🔥 Starting Sathik AI Live Training Loop 🔥")
        
        # Initialize dataset and dataloader
        dataset = WebDataset(data_path, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=2
        )
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Main training loop
        try:
            for epoch in range(num_epochs):
                self.epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                
                # Train epoch
                avg_loss = self.train_epoch(dataloader)
                
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
                
                # Periodic tasks
                if epoch % 10 == 0:
                    self._save_checkpoint()
                
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            logger.info("Training loop ended")
            self._save_checkpoint()  # Final checkpoint
    
    def _run_scheduler(self):
        """Run the periodic task scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# Configuration for the training loop
TRAINING_CONFIG = {
    'vocab_size': 50000,
    'd_model': 2048,
    'num_heads': 32,
    'num_layers': 48,
    'num_experts': 128,
    'top_k': 16,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'batch_size': 8,
    'scheduler_t0': 1000,
    'min_lr': 1e-6
}

# Example usage
if __name__ == "__main__":
    print("🔥 SATHIK AI LIVE TRAINING SYSTEM 🔥")
    print("=" * 50)
    
    # Initialize training loop
    trainer = LiveTrainingLoop(TRAINING_CONFIG)
    
    # Create sample training data if it doesn't exist
    sample_data_path = "sample_training_data.json"
    if not Path(sample_data_path).exists():
        sample_data = [
            {
                'content': f'This is sample training text {i} for the Sathik AI system. '
                          f'It demonstrates how the neural network learns from web data.',
                'metadata': {
                    'source_url': f'https://example.com/page{i}',
                    'timestamp': datetime.now().isoformat(),
                    'quality_score': 0.8
                }
            }
            for i in range(100)
        ]
        
        with open(sample_data_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"Created sample training data: {sample_data_path}")
    
    # Start training
    print("Starting live training loop...")
    trainer.start_training(sample_data_path, num_epochs=10)
    
    print("🔥 SATHIK AI TRAINING COMPLETED! 🔥")

