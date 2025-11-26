import asyncio
import ccxt.async_support as ccxt
import pandas as pd

async def debug_futures():
    print("Connecting to Binance Futures (binanceusdm)...")
    exchange = ccxt.binanceusdm()
    symbol = 'BTC/USDT'
    
    try:
        print(f"Fetching OHLCV for {symbol}...")
        ohlcv = await exchange.fetch_ohlcv(symbol, '1m', limit=10)
        print(f"Fetched {len(ohlcv)} candles.")
        if len(ohlcv) > 0:
            print("Sample:", ohlcv[0])
            
        print(f"Fetching LOB for {symbol}...")
        lob = await exchange.fetch_order_book(symbol, limit=5)
        print("LOB Bids:", len(lob['bids']))
        
    except Exception as e:
        print(f"Error: {e}")
        
    await exchange.close()

if __name__ == "__main__":
    asyncio.run(debug_futures())
