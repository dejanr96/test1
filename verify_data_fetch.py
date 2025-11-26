import asyncio
import logging
from src.data.ingestion import DataIngestionService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataVerifier")

async def main():
    logger.info("Testing Data Fetch (Limit=200,000)...")
    
    data_service = DataIngestionService(exchange_id='binanceusdm', symbol='BTC/USDT:USDT', timeframe='1m')
    await data_service.start_stream() 
    
    try:
        df = await data_service.fetch_ohlcv(limit=200000)
        logger.info(f"Fetched Data Shape: {df.shape}")
        logger.info(f"Start Date: {df.index[0]}")
        logger.info(f"End Date: {df.index[-1]}")
        
        if len(df) < 190000:
            logger.error("FAIL: Fetched fewer than requested!")
        else:
            logger.info("SUCCESS: Data fetch verified.")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await data_service.close()

if __name__ == "__main__":
    asyncio.run(main())
