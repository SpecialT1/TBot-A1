import asyncio
import json
import logging
import os
import sqlite3
import websockets
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color, Rectangle
from kivy.uix.behaviors import TouchRippleBehavior
from solana.rpc.async_api import AsyncClient
from solana.keypair import Keypair
from solders.pubkey import Pubkey
from kivy.clock import Clock
import aiohttp
import base64
from datetime import datetime
import platform

# Setup logging
logging.basicConfig(filename='solana_trader.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Default config
DEFAULT_CONFIG = {
    "solana": {"rpc_endpoint": "https://api.mainnet-beta.solana.com", "priority_fee": 100000},
    "rugcheck": {"bundled_threshold": 50.0, "api_key": ""},
    "solanafm": {"min_score": 80, "api_key": ""},
    "helius": {"api_key": ""},
    "jupiter": {"base_url": "https://quote-api.jup.ag/v6"},
    "trade": {
        "default_amount": 0.1, "slippage_bps_low": 50, "slippage_bps_high": 100,
        "take_profit": 1.3, "dip_threshold": 0.9, "moon_bag_percent": 20,
        "dca_interval": 3600, "dca_amount": 0.05, "limit_order_timeout": 86400,
        "refresh_rate": 5 if platform.system() == "Android" else 1,
        "sync_interval": 15,
        "max_graphs": 5,
        "max_points": 50,
        "zoom_sensitivity": 0.1
    },
    "copy_trading": {
        "wallets": [],
        "trade_size": 0.1,
        "delay": 5,
        "min_liquidity": 10000
    },
    "filters": {"min_price": 0.0001, "min_volume": 10000, "min_liquidity": 5000, "min_market_cap": 100000}
}
CONFIG_FILE = 'config.json'
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(DEFAULT_CONFIG, f)
CONFIG = json.load(open(CONFIG_FILE, 'r'))

# Global state
PRICE_FEED = {}
WALLET = None
SOLANA_CLIENT = AsyncClient(CONFIG['solana']['rpc_endpoint'])
HOLDINGS = {}
LIMIT_ORDERS = {}
DCA_TASKS = {}
COPY_TRADING = {}
RUNNING = False
PRICE_TASK = None
SYNC_TASK = None

# Database setup
def init_db():
    conn = sqlite3.connect('solana_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS tokens
                 (id INTEGER PRIMARY KEY, token_address TEXT, name TEXT, price REAL, volume REAL, 
                  liquidity REAL, market_cap REAL, timestamp TEXT, status TEXT, dev_address TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS blacklists
                 (type TEXT, address TEXT, reason TEXT, UNIQUE(type, address))''')
    c.execute('''CREATE TABLE IF NOT EXISTS pl_history
                 (id INTEGER PRIMARY KEY, timestamp TEXT, token_address TEXT, pl_value REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS transactions
                 (id INTEGER PRIMARY KEY, token_address TEXT, tx_hash TEXT, action TEXT, price REAL, 
                  quantity REAL, timestamp TEXT, UNIQUE(tx_hash))''')
    conn.commit()
    conn.close()

# Custom Widget for P/L Graph with Zoom and Reset
class PLGraphWidget(TouchRippleBehavior, Widget):
    def __init__(self, token_address, **kwargs):
        super().__init__(**kwargs)
        self.token_address = token_address
        self.zoom = 1.0
        self.offset_x = 0
        self.touch_pos = None
        self.reset_btn = Button(text="Reset Zoom", size_hint=(None, None), size=(100, 30), pos=(self.x, self.y))
        self.reset_btn.bind(on_press=self.reset_zoom)
        self.add_widget(self.reset_btn)
        self.bind(size=self.update_graph, pos=self.update_graph)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.touch_pos = touch.pos
            self.ripple_show(touch)
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.touch_pos and self.collide_point(*touch.pos):
            if touch.is_double_tap:  # Pinch zoom detection
                return True
            dx = touch.pos[0] - self.touch_pos[0]
            self.offset_x += dx / self.zoom
            self.touch_pos = touch.pos
            self.update_graph()
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            self.ripple_fade()
            self.touch_pos = None
            return True
        return super().on_touch_up(touch)

    def on_touch_scroll(self, touch):
        if self.collide_point(*touch.pos):
            zoom_factor = 1 + CONFIG['trade']['zoom_sensitivity'] * touch.dwheel
            self.zoom = max(0.5, min(5.0, self.zoom * zoom_factor))  # Limit zoom range
            self.update_graph()
            return True
        return super().on_touch_scroll(touch)

    def reset_zoom(self, instance):
        self.zoom = 1.0
        self.offset_x = 0
        self.update_graph()

    def update_graph(self, *args):
        self.canvas.clear()
        self.reset_btn.pos = (self.x + self.width - 110, self.y + self.height - 40)  # Position button top-right
        conn = sqlite3.connect('solana_data.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, pl_value FROM pl_history WHERE token_address = ? ORDER BY timestamp DESC LIMIT ?",
                  (self.token_address, CONFIG['trade']['max_points']))
        data = c.fetchall()
        conn.close()
        
        if not data or len(data) < 2:
            return
        
        timestamps, pl_values = zip(*data[::-1])
        min_pl, max_pl = min(pl_values), max(pl_values)
        if min_pl == max_pl:
            min_pl -= 1
            max_pl += 1
        
        with self.canvas:
            Color(1, 1, 1, 1)  # White background
            Rectangle(pos=self.pos, size=self.size)
            
            Color(0, 0, 0, 1)  # Black axes
            Line(points=[self.x, self.y, self.x + self.width, self.y], width=1)  # X-axis
            Line(points=[self.x, self.y, self.x, self.y + self.height], width=1)  # Y-axis
            
            Color(0, 1, 0, 1)  # Green line
            x_step = (self.width * 0.9 / (len(timestamps) - 1)) * self.zoom
            y_scale = self.height * 0.8 / (max_pl - min_pl)
            points = []
            for i, (ts, pl) in enumerate(zip(timestamps, pl_values)):
                x = self.x + i * x_step + self.width * 0.05 - self.offset_x
                y = self.y + (pl - min_pl) * y_scale + self.height * 0.1
                if self.x <= x <= self.x + self.width:  # Clip to visible area
                    points.extend([x, y])
            Line(points=points, width=1)
            
            # Axis Labels
            Color(0, 0, 0, 1)
            self.add_label("Time", self.x + self.width / 2, self.y - 20, center_x=True)
            self.add_label("P/L (SOL)", self.x - 40, self.y + self.height / 2, rotate=90, center_y=True)
            self.add_label(f"{min_pl:.2f}", self.x - 20, self.y + self.height * 0.1, center_y=True)
            self.add_label(f"{max_pl:.2f}", self.x - 20, self.y + self.height * 0.9, center_y=True)
            visible_start = max(0, int(self.offset_x / x_step))
            visible_end = min(len(timestamps) - 1, int((self.offset_x + self.width) / x_step))
            if visible_start < len(timestamps):
                self.add_label(timestamps[visible_start][:10], self.x + self.width * 0.05, self.y - 20, center_x=True)
            if visible_end < len(timestamps):
                self.add_label(timestamps[visible_end][:10], self.x + self.width * 0.95, self.y - 20, center_x=True)

    def add_label(self, text, x, y, center_x=False, center_y=False, rotate=0):
        label = Label(text=text, font_size=12, color=(0, 0, 0, 1), pos=(x, y))
        if center_x:
            label.x -= label.texture_size[0] / 2
        if center_y:
            label.y -= label.texture_size[1] / 2
        if rotate:
            label.canvas.before.add(Color(0, 0, 0, 1))
            label.canvas.before.add(PushMatrix())
            label.canvas.before.add(Rotate(angle=rotate, origin=(label.center_x, label.center_y)))
            label.canvas.after.add(PopMatrix())
        self.canvas.add(label.canvas)

class SolanaTraderApp(App):
    def __init__(self):
        super().__init__()
        self.loop = asyncio.get_event_loop()
        self.log_messages = []

    def build(self):
        self.root = TabbedPanel()
        self.root.default_tab_text = "Setup"
        
        # Setup Tab
        setup_tab = BoxLayout(orientation='vertical', padding=10)
        setup_tab.add_widget(Label(text="Wallet Private Key:"))
        self.wallet_input = TextInput(multiline=False, password=True)
        setup_tab.add_widget(self.wallet_input)
        
        setup_tab.add_widget(Label(text="Rugcheck API Key:"))
        self.rugcheck_input = TextInput(multiline=False, password=True)
        setup_tab.add_widget(self.rugcheck_input)
        
        setup_tab.add_widget(Label(text="SolanaFM API Key:"))
        self.solanafm_input = TextInput(multiline=False, password=True)
        setup_tab.add_widget(self.solanafm_input)
        
        setup_tab.add_widget(Label(text="Helius API Key:"))
        self.helius_input = TextInput(multiline=False, password=True)
        setup_tab.add_widget(self.helius_input)
        
        start_btn = Button(text="Start Trading")
        start_btn.bind(on_press=self.start_trading)
        setup_tab.add_widget(start_btn)
        
        self.root.default_tab_content = setup_tab
        
        # Trade Tab
        trade_tab = BoxLayout(orientation='vertical', padding=10)
        trade_tab.add_widget(Label(text="Token Address:"))
        self.trade_token = TextInput(multiline=False)
        trade_tab.add_widget(self.trade_token)
        trade_btn = Button(text="Trade")
        trade_btn.bind(on_press=lambda x: self.loop.create_task(self.trade(self.trade_token.text)))
        trade_tab.add_widget(trade_btn)
        self.root.add_widget(TabbedPanelItem(text="Trade", content=trade_tab))
        
        # Snipe Tab
        snipe_tab = BoxLayout(orientation='vertical', padding=10)
        snipe_tab.add_widget(Label(text="Token Address:"))
        self.snipe_token = TextInput(multiline=False)
        snipe_tab.add_widget(self.snipe_token)
        snipe_btn = Button(text="Snipe")
        snipe_btn.bind(on_press=lambda x: self.loop.create_task(self.snipe(self.snipe_token.text)))
        snipe_tab.add_widget(snipe_btn)
        self.root.add_widget(TabbedPanelItem(text="Snipe", content=snipe_tab))
        
        # Copy Tab
        copy_tab = GridLayout(cols=2, padding=10)
        copy_tab.add_widget(Label(text="Wallet:"))
        self.copy_wallet = TextInput(multiline=False)
        copy_tab.add_widget(self.copy_wallet)
        copy_tab.add_widget(Label(text="Trade Size (SOL):"))
        self.copy_size = TextInput(multiline=False, text=str(CONFIG['copy_trading']['trade_size']))
        copy_tab.add_widget(self.copy_size)
        copy_tab.add_widget(Label(text="Delay (s):"))
        self.copy_delay = TextInput(multiline=False, text=str(CONFIG['copy_trading']['delay']))
        copy_tab.add_widget(self.copy_delay)
        copy_btn = Button(text="Add")
        copy_btn.bind(on_press=lambda x: self.loop.create_task(self.copy(self.copy_wallet.text)))
        copy_tab.add_widget(copy_btn)
        self.root.add_widget(TabbedPanelItem(text="Copy", content=copy_tab))
        
        # Limit Tab
        limit_tab = GridLayout(cols=2, padding=10)
        limit_tab.add_widget(Label(text="Token:"))
        self.limit_token = TextInput(multiline=False)
        limit_tab.add_widget(self.limit_token)
        limit_tab.add_widget(Label(text="Price:"))
        self.limit_price = TextInput(multiline=False)
        limit_tab.add_widget(self.limit_price)
        limit_tab.add_widget(Label(text="Action:"))
        self.limit_action = TextInput(multiline=False)
        limit_tab.add_widget(self.limit_action)
        limit_tab.add_widget(Label(text="Amount:"))
        self.limit_amount = TextInput(multiline=False)
        limit_tab.add_widget(self.limit_amount)
        limit_btn = Button(text="Set Limit")
        limit_btn.bind(on_press=lambda x: self.loop.create_task(self.limit(self.limit_token.text, self.limit_price.text, self.limit_action.text, self.limit_amount.text)))
        limit_tab.add_widget(limit_btn)
        self.root.add_widget(TabbedPanelItem(text="Limit", content=limit_tab))
        
        # DCA Tab
        dca_tab = BoxLayout(orientation='vertical', padding=10)
        dca_tab.add_widget(Label(text="Token Address:"))
        self.dca_token = TextInput(multiline=False)
        dca_tab.add_widget(self.dca_token)
        dca_btn = Button(text="Start DCA")
        dca_btn.bind(on_press=lambda x: self.loop.create_task(self.dca(self.dca_token.text)))
        dca_tab.add_widget(dca_btn)
        self.root.add_widget(TabbedPanelItem(text="DCA", content=dca_tab))
        
        # Settings Tab
        settings_tab = GridLayout(cols=2, padding=10)
        settings_tab.add_widget(Label(text="Default Amount (SOL):"))
        self.default_amount = TextInput(multiline=False, text=str(CONFIG['trade']['default_amount']))
        settings_tab.add_widget(self.default_amount)
        settings_tab.add_widget(Label(text="Slippage Low (bps):"))
        self.slippage_low = TextInput(multiline=False, text=str(CONFIG['trade']['slippage_bps_low']))
        settings_tab.add_widget(self.slippage_low)
        settings_tab.add_widget(Label(text="Slippage High (bps):"))
        self.slippage_high = TextInput(multiline=False, text=str(CONFIG['trade']['slippage_bps_high']))
        settings_tab.add_widget(self.slippage_high)
        settings_tab.add_widget(Label(text="Take Profit (%):"))
        self.take_profit = TextInput(multiline=False, text=str(CONFIG['trade']['take_profit']))
        settings_tab.add_widget(self.take_profit)
        settings_tab.add_widget(Label(text="Dip Threshold (%):"))
        self.dip_threshold = TextInput(multiline=False, text=str(CONFIG['trade']['dip_threshold']))
        settings_tab.add_widget(self.dip_threshold)
        settings_tab.add_widget(Label(text="Moon Bag (%):"))
        self.moon_bag = TextInput(multiline=False, text=str(CONFIG['trade']['moon_bag_percent']))
        settings_tab.add_widget(self.moon_bag)
        settings_tab.add_widget(Label(text="DCA Amount (SOL):"))
        self.dca_amount = TextInput(multiline=False, text=str(CONFIG['trade']['dca_amount']))
        settings_tab.add_widget(self.dca_amount)
        settings_tab.add_widget(Label(text="DCA Interval (s):"))
        self.dca_interval = TextInput(multiline=False, text=str(CONFIG['trade']['dca_interval']))
        settings_tab.add_widget(self.dca_interval)
        settings_tab.add_widget(Label(text="Refresh Rate (s):"))
        self.refresh_rate = DropDown()
        rates = [0.5, 1, 5, 10, 30] if platform.system() != "Android" else [1, 5, 10, 30]
        for rate in rates:
            btn = Button(text=str(rate), size_hint_y=None, height=44)
            btn.bind(on_release=lambda btn: self.refresh_rate.select(btn.text))
            self.refresh_rate.add_widget(btn)
        self.refresh_btn = Button(text=str(CONFIG['trade']['refresh_rate']))
        self.refresh_btn.bind(on_release=self.refresh_rate.open)
        self.refresh_rate.bind(on_select=lambda instance, x: setattr(self.refresh_btn, 'text', x))
        settings_tab.add_widget(self.refresh_btn)
        save_btn = Button(text="Save Settings")
        save_btn.bind(on_press=self.save_settings)
        settings_tab.add_widget(save_btn)
        self.root.add_widget(TabbedPanelItem(text="Settings", content=settings_tab))
        
        # Holdings Tab with Manual Sync Button
        holdings_tab = BoxLayout(orientation='vertical', padding=10)
        self.sync_btn = Button(text="Manual Sync")
        self.sync_btn.bind(on_press=lambda x: self.loop.create_task(self.sync_holdings()))
        holdings_tab.add_widget(self.sync_btn)
        holdings_scroll = ScrollView()
        self.holdings_grid = GridLayout(cols=6, padding=10, size_hint_y=None)
        self.holdings_grid.bind(minimum_height=self.holdings_grid.setter('height'))
        self.holdings_grid.add_widget(Label(text="Token"))
        self.holdings_grid.add_widget(Label(text="Quantity"))
        self.holdings_grid.add_widget(Label(text="Value (SOL)"))
        self.holdings_grid.add_widget(Label(text="Avg Buy Price"))
        self.holdings_grid.add_widget(Label(text="P/L (%)"))
        self.holdings_grid.add_widget(Label(text="Action"))
        holdings_scroll.add_widget(self.holdings_grid)
        holdings_tab.add_widget(holdings_scroll)
        self.root.add_widget(TabbedPanelItem(text="Holdings", content=holdings_tab))
        
        # P/L History Tab
        pl_tab = ScrollView()
        self.pl_grid = GridLayout(cols=1, padding=10, size_hint_y=None)
        self.pl_grid.bind(minimum_height=self.pl_grid.setter('height'))
        
        total_pl_box = BoxLayout(orientation='vertical')
        total_pl_box.add_widget(Label(text="Total P/L History (Zoom: Pinch/Scroll)"))
        self.total_pl_graph = PLGraphWidget(token_address="TOTAL", size_hint=(1, None), height=200)
        total_pl_box.add_widget(self.total_pl_graph)
        self.pl_grid.add_widget(total_pl_box)
        
        pl_tab.add_widget(self.pl_grid)
        self.root.add_widget(TabbedPanelItem(text="P/L History", content=pl_tab))
        
        # Log Tab with Pause Button
        log_tab = BoxLayout(orientation='vertical', padding=10)
        self.pause_btn = Button(text="Pause")
        self.pause_btn.bind(on_press=self.toggle_pause)
        log_tab.add_widget(self.pause_btn)
        log_scroll = ScrollView()
        self.log_area = Label(text="", halign="left", valign="top", size_hint_y=None)
        self.log_area.bind(texture_size=self.log_area.setter('size'))
        log_scroll.add_widget(self.log_area)
        log_tab.add_widget(log_scroll)
        self.root.add_widget(TabbedPanelItem(text="Log", content=log_tab))
        
        Clock.schedule_interval(self.update_log, 1)
        self.update_holdings_task = Clock.schedule_interval(self.update_holdings, CONFIG['trade']['refresh_rate'])
        self.update_pl_task = Clock.schedule_interval(self.update_pl_graphs, CONFIG['trade']['refresh_rate'])
        return self.root

    def start_trading(self, instance):
        global WALLET, RUNNING, PRICE_TASK, SYNC_TASK
        try:
            WALLET = Keypair.from_secret_key(bytes.fromhex(self.wallet_input.text))
            CONFIG['solana']['private_key'] = self.wallet_input.text
            CONFIG['rugcheck']['api_key'] = self.rugcheck_input.text
            CONFIG['solanafm']['api_key'] = self.solanafm_input.text
            CONFIG['helius']['api_key'] = self.helius_input.text
            with open(CONFIG_FILE, 'w') as f:
                json.dump(CONFIG, f)
            RUNNING = True
            PRICE_TASK = self.loop.create_task(fetch_real_time_prices())
            SYNC_TASK = self.loop.create_task(self.sync_holdings_periodic())
            self.loop.create_task(self.sync_holdings())
            self.log("Bot started successfully!")
        except Exception as e:
            self.log(f"Error starting bot: {e}")

    def toggle_pause(self, instance):
        global RUNNING, PRICE_TASK, SYNC_TASK
        RUNNING = not RUNNING
        if not RUNNING:
            if PRICE_TASK:
                PRICE_TASK.cancel()
                PRICE_TASK = None
            if SYNC_TASK:
                SYNC_TASK.cancel()
                SYNC_TASK = None
            for task in DCA_TASKS.values():
                task.cancel()
            DCA_TASKS.clear()
            self.log("Bot paused - power saving enabled.")
        elif RUNNING:
            PRICE_TASK = self.loop.create_task(fetch_real_time_prices())
            SYNC_TASK = self.loop.create_task(self.sync_holdings_periodic())
            self.loop.create_task(self.sync_holdings())
            self.log("Bot resumed.")
        self.pause_btn.text = "Resume" if not RUNNING else "Pause"

    def save_settings(self, instance):
        CONFIG['trade']['default_amount'] = float(self.default_amount.text)
        CONFIG['trade']['slippage_bps_low'] = int(self.slippage_low.text)
        CONFIG['trade']['slippage_bps_high'] = int(self.slippage_high.text)
        CONFIG['trade']['take_profit'] = float(self.take_profit.text)
        CONFIG['trade']['dip_threshold'] = float(self.dip_threshold.text)
        CONFIG['trade']['moon_bag_percent'] = float(self.moon_bag.text)
        CONFIG['trade']['dca_amount'] = float(self.dca_amount.text)
        CONFIG['trade']['dca_interval'] = int(self.dca_interval.text)
        CONFIG['trade']['refresh_rate'] = float(self.refresh_btn.text)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f)
        self.update_holdings_task.unschedule()
        self.update_pl_task.unschedule()
        self.update_holdings_task = Clock.schedule_interval(self.update_holdings, CONFIG['trade']['refresh_rate'])
        self.update_pl_task = Clock.schedule_interval(self.update_pl_graphs, CONFIG['trade']['refresh_rate'])
        self.log("Settings saved!")

    def update_log(self, dt):
        self.log_area.text = "\n".join(self.log_messages[-20:])

    async def sync_holdings(self):
        global HOLDINGS
        if not WALLET: return
        url = f"https://api.helius.xyz/v0/addresses/{str(WALLET.public_key)}/balances"
        params = {"api-key": CONFIG['helius']['api_key']}
        retries = 3
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=5) as resp:
                        data = await resp.json()
                        HOLDINGS.clear()
                        for token in data.get('tokens', []):
                            token_address = token.get('mint')
                            amount = token.get('amount', 0) / 10**token.get('decimals', 6)
                            if amount > 0:
                                HOLDINGS[token_address] = amount
                        total_pl = await self.calculate_total_pl()
                        conn = sqlite3.connect('solana_data.db')
                        c = conn.cursor()
                        c.execute("INSERT INTO pl_history (timestamp, token_address, pl_value) VALUES (?, ?, ?)",
                                  (datetime.now().isoformat(), "TOTAL", total_pl))
                        for token_address in HOLDINGS:
                            pl = await self.calculate_token_pl(token_address)
                            c.execute("INSERT INTO pl_history (timestamp, token_address, pl_value) VALUES (?, ?, ?)",
                                      (datetime.now().isoformat(), token_address, pl))
                        conn.commit()
                        conn.close()
                        await self.log(f"Holdings synced: {len(HOLDINGS)} tokens, Total P/L: {total_pl:.2f} SOL")
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                await self.log(f"Sync attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
        await self.log("Failed to sync holdings after retries")

    async def sync_holdings_periodic(self):
        while RUNNING:
            await self.sync_holdings()
            await asyncio.sleep(CONFIG['trade']['sync_interval'])

    async def calculate_total_pl(self):
        total_pl = 0
        for token_address, quantity in HOLDINGS.items():
            current_price = await self.get_accurate_price(token_address)
            buy_prices = await self.get_exact_buy_prices(token_address)
            if not buy_prices:
                continue
            total_cost = sum(price * qty for price, qty in buy_prices.values())
            avg_buy_price = total_cost / sum(buy_prices.values())
            pl = (current_price - avg_buy_price) * quantity if current_price else 0
            total_pl += pl
        return total_pl

    async def calculate_token_pl(self, token_address):
        current_price = await self.get_accurate_price(token_address)
        quantity = HOLDINGS.get(token_address, 0)
        buy_prices = await self.get_exact_buy_prices(token_address)
        if not buy_prices:
            return 0
        total_cost = sum(price * qty for price, qty in buy_prices.values())
        avg_buy_price = total_cost / sum(buy_prices.values())
        return (current_price - avg_buy_price) * quantity if current_price else 0

    async def get_accurate_price(self, token_address):
        jupiter_price = PRICE_FEED.get(token_address, 0)
        solanafm_price = await self.fetch_solanafm_price(token_address)
        return jupiter_price if jupiter_price else solanafm_price or 0

    async def fetch_solanafm_price(self, token_address):
        url = f"https://api.solana.fm/v1/tokens/{token_address}"
        headers = {"Authorization": f"Bearer {CONFIG['solanafm']['api_key']}"}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=5) as resp:
                    data = await resp.json()
                    return data.get('price', 0)
        except Exception:
            return 0

    async def get_exact_buy_prices(self, token_address):
        conn = sqlite3.connect('solana_data.db')
        c = conn.cursor()
        c.execute("SELECT tx_hash, price, quantity FROM transactions WHERE token_address = ? AND action = 'buy'",
                  (token_address,))
        tx_data = c.fetchall()
        conn.close()
        
        buy_prices = {}
        for tx_hash, price, quantity in tx_data:
            buy_prices[tx_hash] = (price, quantity)
        
        url = f"https://api.helius.xyz/v0/transactions?account={str(WALLET.public_key)}&limit=100"
        params = {"api-key": CONFIG['helius']['api_key']}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                for tx in data.get('transactions', []):
                    tx_hash = tx['signature']
                    if tx_hash in buy_prices:
                        continue
                    for transfer in tx.get('token_transfers', []):
                        if transfer['mint'] == token_address and transfer['to'] == str(WALLET.public_key):
                            price = await self.get_accurate_price(token_address)  # Use current price as fallback
                            quantity = transfer['amount'] / 10**transfer.get('decimals', 6)
                            buy_prices[tx_hash] = (price, quantity)
                            c = conn.cursor()
                            c.execute("INSERT OR IGNORE INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                                      (token_address, tx_hash, 'buy', price, quantity, datetime.now().isoformat()))
                            conn.commit()
        conn.close()
        return buy_prices

    def update_holdings(self, dt):
        if not RUNNING: return
        self.holdings_grid.clear_widgets(children=self.holdings_grid.children[6:])
        conn = sqlite3.connect('solana_data.db')
        c = conn.cursor()
        for token_address, quantity in HOLDINGS.items():
            current_price = PRICE_FEED.get(token_address, 0)
            value_sol = quantity * current_price if current_price else 0
            
            c.execute("SELECT SUM(price * quantity) / SUM(quantity) FROM transactions WHERE token_address = ? AND action = 'buy'",
                      (token_address,))
            avg_buy_price = c.fetchone()[0] or current_price or 0
            pl_percent = ((current_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price and current_price else 0
            
            self.holdings_grid.add_widget(Label(text=token_address[:8]))
            self.holdings_grid.add_widget(Label(text=f"{quantity:.4f}"))
            self.holdings_grid.add_widget(Label(text=f"{value_sol:.4f}"))
            self.holdings_grid.add_widget(Label(text=f"{avg_buy_price:.6f}"))
            self.holdings_grid.add_widget(Label(text=f"{pl_percent:.2f}%"))
            sell_btn = Button(text="Sell")
            sell_btn.bind(on_press=lambda x, t=token_address: self.loop.create_task(self.sell_holding(t)))
            self.holdings_grid.add_widget(sell_btn)
        conn.close()

    def update_pl_graphs(self, dt):
        if not RUNNING: return
        # Total P/L Graph updated via PLGraphWidget
        
        # Per-Token P/L Graphs (limited to top 5 by value)
        self.pl_grid.clear_widgets(children=self.pl_grid.children[1:])
        token_values = [(token, qty * PRICE_FEED.get(token, 0)) for token, qty in HOLDINGS.items()]
        token_values.sort(key=lambda x: x[1], reverse=True)
        top_tokens = [t[0] for t in token_values[:CONFIG['trade']['max_graphs']]]
        
        for token_address in top_tokens:
            token_box = BoxLayout(orientation='vertical')
            token_box.add_widget(Label(text=f"P/L History: {token_address[:8]} (Zoom: Pinch/Scroll)"))
            graph = PLGraphWidget(token_address=token_address, size_hint=(1, None), height=200)
            token_box.add_widget(graph)
            self.pl_grid.add_widget(token_box)

    async def sell_holding(self, token_address):
        if not RUNNING or token_address not in HOLDINGS: return
        amount = HOLDINGS[token_address] * (1 - CONFIG['trade']['moon_bag_percent'] / 100) * PRICE_FEED.get(token_address, 0)
        success, tx_hash = await execute_jupiter_trade(token_address, "sell", amount, 50000)
        if success:
            await self.log(f"SELL {amount:.4f} SOL of {token_address} - Tx: {tx_hash} (Moon bag: {HOLDINGS[token_address] * (CONFIG['trade']['moon_bag_percent'] / 100):.4f} tokens retained)")
            conn = sqlite3.connect('solana_data.db')
            c = conn.cursor()
            c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                      (token_address, tx_hash, 'sell', PRICE_FEED.get(token_address, 0), amount / PRICE_FEED.get(token_address, 0), datetime.now().isoformat()))
            conn.commit()
            conn.close()
            await self.sync_holdings()
        else:
            await self.log(f"Sell failed: {tx_hash}")

    async def log(self, message):
        self.log_messages.append(message)
        logging.info(message)

    async def trade(self, token_address):
        if not RUNNING or not await self.validate_token(token_address): return
        price = await self.get_accurate_price(token_address)
        if price == 0:
            await self.log("Waiting for price data...")
            return
        
        conn = sqlite3.connect('solana_data.db')
        c = conn.cursor()
        c.execute("SELECT price FROM tokens WHERE token_address = ? ORDER BY timestamp DESC LIMIT 1", (token_address,))
        row = c.fetchone()
        prev_price = row[0] if row else price
        
        action, amount = None, CONFIG['trade']['default_amount']
        liquidity = 50000
        
        if price < CONFIG['trade']['dip_threshold'] * prev_price:
            action = "buy"
            success, tx_hash = await execute_jupiter_trade(token_address, "buy", amount, liquidity)
            if success:
                await self.log(f"BUY {amount:.4f} SOL of {token_address} - Tx: {tx_hash}")
                c.execute("INSERT INTO tokens (token_address, name, price, timestamp, status, dev_address) VALUES (?, ?, ?, ?, ?, ?)",
                          (token_address, token_address[:8], price, datetime.now().isoformat(), "tracked", "unknown"))
                c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                          (token_address, tx_hash, 'buy', price, amount / price, datetime.now().isoformat()))
        elif price > await self.get_avg_buy_price(token_address) * CONFIG['trade']['take_profit'] and token_address in HOLDINGS:
            action = "sell"
            sell_amount = HOLDINGS[token_address] * (1 - CONFIG['trade']['moon_bag_percent'] / 100)
            amount = sell_amount * price
            success, tx_hash = await execute_jupiter_trade(token_address, action, amount, liquidity)
            if success:
                await self.log(f"SELL {amount:.4f} SOL of {token_address} - Tx: {tx_hash} (Moon bag: {HOLDINGS[token_address] * (CONFIG['trade']['moon_bag_percent'] / 100):.4f} tokens retained)")
                c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                          (token_address, tx_hash, 'sell', price, amount / price, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        if action and success:
            await self.sync_holdings()

    async def get_avg_buy_price(self, token_address):
        buy_prices = await self.get_exact_buy_prices(token_address)
        if not buy_prices:
            return await self.get_accurate_price(token_address)
        total_cost = sum(price * qty for price, qty in buy_prices.values())
        total_qty = sum(buy_prices.values())
        return total_cost / total_qty if total_qty else 0

    async def snipe(self, token_address):
        if not RUNNING or not await self.validate_token(token_address): return
        amount = CONFIG['trade']['default_amount']
        success, tx_hash = await execute_jupiter_trade(token_address, "buy", amount, 50000)
        if success:
            await self.log(f"SNIPE BUY {amount:.4f} SOL of {token_address} - Tx: {tx_hash}")
            price = await self.get_accurate_price(token_address)
            conn = sqlite3.connect('solana_data.db')
            c = conn.cursor()
            c.execute("INSERT INTO tokens (token_address, name, price, timestamp, status, dev_address) VALUES (?, ?, ?, ?, ?, ?)",
                      (token_address, token_address[:8], price, datetime.now().isoformat(), "tracked", "unknown"))
            c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                      (token_address, tx_hash, 'buy', price, amount / price, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            await self.sync_holdings()

    async def copy(self, wallet):
        if not RUNNING: return
        CONFIG['copy_trading']['wallets'].append(wallet)
        CONFIG['copy_trading']['trade_size'] = float(self.copy_size.text)
        CONFIG['copy_trading']['delay'] = int(self.copy_delay.text)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f)
        await self.log(f"Copy-trading wallet {wallet} added with size {CONFIG['copy_trading']['trade_size']} SOL, delay {CONFIG['copy_trading']['delay']}s")

    async def limit(self, token_address, price_target, action, amount):
        if not RUNNING or not await self.validate_token(token_address): return
        LIMIT_ORDERS[token_address] = (float(price_target), action.lower(), float(amount), CONFIG['telegram']['chat_id'], datetime.now().timestamp() + CONFIG['trade']['limit_order_timeout'])
        await self.log(f"Limit {action} order set for {token_address} at {price_target}")

    async def dca(self, token_address):
        if not RUNNING or not await self.validate_token(token_address) or token_address in DCA_TASKS: return
        DCA_TASKS[token_address] = self.loop.create_task(self.run_dca(token_address))
        await self.log(f"DCA started for {token_address}")

    async def validate_token(self, token_address):
        score, details = await check_solanafm_score(token_address)
        if score < CONFIG['solanafm']['min_score']:
            await self.log(f"Alert: {token_address} SolanaFM score {score} (<80)! {details}")
            blacklist_entity('token', token_address, f"Low SolanaFM score: {score}")
            return False
        
        is_good, is_bundled, dev_address, rug_details = await verify_with_rugcheck(token_address)
        if not is_good or is_bundled:
            blacklist_entity('token', token_address, f"Rugcheck: {rug_details}" if not is_good else "Bundled supply")
            await self.log(f"Token {token_address} not safe!")
            return False
        return True

    async def run_dca(self, token_address):
        while RUNNING and token_address in DCA_TASKS:
            amount = CONFIG['trade']['dca_amount']
            success, tx_hash = await execute_jupiter_trade(token_address, "buy", amount, 50000)
            if success:
                await self.log(f"DCA BUY {amount:.4f} SOL of {token_address} - Tx: {tx_hash}")
                price = await self.get_accurate_price(token_address)
                conn = sqlite3.connect('solana_data.db')
                c = conn.cursor()
                c.execute("INSERT INTO tokens (token_address, name, price, timestamp, status, dev_address) VALUES (?, ?, ?, ?, ?, ?)",
                          (token_address, token_address[:8], price, datetime.now().isoformat(), "tracked", "unknown"))
                c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                          (token_address, tx_hash, 'buy', price, amount / price, datetime.now().isoformat()))
                conn.commit()
                conn.close()
                await self.sync_holdings()
            await asyncio.sleep(CONFIG['trade']['dca_interval'])

# Trading functions
async def fetch_real_time_prices():
    uri = "wss://cache.jup.ag/prices"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"command": "subscribe", "channel": "prices"}))
        while RUNNING:
            try:
                data = json.loads(await ws.recv())
                if 'token_address' in data and 'price' in data:
                    PRICE_FEED[data['token_address']] = float(data['price'])
                    await check_limit_orders()
                    await check_copy_trading()
            except asyncio.CancelledError:
                break

async def verify_with_rugcheck(token_address):
    url = f"https://api.rugcheck.xyz/v1/tokens/{token_address}/report"
    headers = {"Authorization": f"Bearer {CONFIG['rugcheck']['api_key']}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=5) as resp:
            data = await resp.json()
            is_good = data.get('status', '').lower() == 'good'
            is_bundled = any(h.get('percentage', 0) > CONFIG['rugcheck']['bundled_threshold'] 
                             for h in data.get('top_holders', []))
            return is_good, is_bundled, data.get('creator', 'unknown'), data.get('details', '')

async def check_solanafm_score(token_address):
    url = f"https://api.solana.fm/v1/tokens/{token_address}"
    headers = {"Authorization": f"Bearer {CONFIG['solanafm']['api_key']}"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=5) as resp:
                data = await resp.json()
                liquidity = data.get('liquidity', 0)
                market_cap = data.get('market_cap', 1)
                score = min(100, int((liquidity / market_cap) * 1000))
                return score, f"Liquidity: {liquidity}, Market Cap: {market_cap}"
    except Exception as e:
        logging.error(f"SolanaFM failed: {e}")
        return 0, "API error"

async def execute_jupiter_trade(token_address, action, amount, liquidity):
    input_mint = "So11111111111111111111111111111111111111112"
    output_mint = token_address if action == "buy" else input_mint
    input_mint = input_mint if action == "buy" else token_address
    amount_lamports = int(amount * 10**9)
    slippage_bps = CONFIG['trade']['slippage_bps_low'] if liquidity > 100000 else CONFIG['trade']['slippage_bps_high']
    
    quote_url = f"{CONFIG['jupiter']['base_url']}/quote?inputMint={input_mint}&outputMint={output_mint}&amount={amount_lamports}&slippageBps={slippage_bps}"
    async with aiohttp.ClientSession() as session:
        async with session.get(quote_url) as resp:
            quote = await resp.json()
            if 'error' in quote:
                return False, quote['error']
        
        swap_url = f"{CONFIG['jupiter']['base_url']}/swap"
        payload = {
            "quoteResponse": quote,
            "userPublicKey": str(WALLET.public_key),
            "wrapAndUnwrapSol": True,
            "prioritizationFeeLamports": CONFIG['solana']['priority_fee']
        }
        async with session.post(swap_url, json=payload) as resp:
            swap_data = await resp.json()
            if 'swapTransaction' not in swap_data:
                return False, swap_data.get('error', 'Swap failed')
            
            from solders.transaction import Transaction as SoldersTx
            tx = SoldersTx.from_bytes(base64.b64decode(swap_data['swapTransaction']))
            tx.sign([WALLET])
            tx_resp = await SOLANA_CLIENT.send_transaction(tx)
            return True, tx_resp.value

async def check_copy_trading():
    if not RUNNING: return
    for wallet in CONFIG['copy_trading']['wallets']:
        url = "https://api.helius.xyz/v0/transactions"
        params = {"api-key": CONFIG['helius']['api_key'], "account": wallet, "limit": 1}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()
                txs = data.get('transactions', [])
                if txs and txs[0]['signature'] not in COPY_TRADING.get(wallet, []):
                    COPY_TRADING.setdefault(wallet, []).append(txs[0]['signature'])
                    token_address = txs[0].get('token_transfers', [{}])[0].get('mint', None)
                    if token_address and await app.validate_token(token_address):
                        liquidity = (await check_solanafm_score(token_address))[0] * 1000
                        if liquidity >= CONFIG['copy_trading']['min_liquidity']:
                            await asyncio.sleep(CONFIG['copy_trading']['delay'])
                            success, tx_hash = await execute_jupiter_trade(token_address, "buy", CONFIG['copy_trading']['trade_size'], liquidity)
                            if success:
                                await app.log(f"COPY BUY {CONFIG['copy_trading']['trade_size']} SOL of {token_address} - Tx: {tx_hash}")
                                price = await app.get_accurate_price(token_address)
                                conn = sqlite3.connect('solana_data.db')
                                c = conn.cursor()
                                c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                                          (token_address, tx_hash, 'buy', price, CONFIG['copy_trading']['trade_size'] / price, datetime.now().isoformat()))
                                conn.commit()
                                conn.close()
                                await app.sync_holdings()

async def check_limit_orders():
    if not RUNNING: return
    conn = sqlite3.connect('solana_data.db')
    c = conn.cursor()
    for token_address, (price_target, action, amount, chat_id, expiry) in list(LIMIT_ORDERS.items()):
        current_price = await app.get_accurate_price(token_address)
        if current_price and ((action == "buy" and current_price <= price_target) or (action == "sell" and current_price >= price_target)):
            success, tx_hash = await execute_jupiter_trade(token_address, action, amount, 50000)
            if success:
                if action == "buy":
                    HOLDINGS[token_address] = HOLDINGS.get(token_address, 0) + (amount / current_price)
                    c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                              (token_address, tx_hash, 'buy', current_price, amount / current_price, datetime.now().isoformat()))
                elif action == "sell":
                    HOLDINGS[token_address] = max(0, HOLDINGS[token_address] - (amount / current_price))
                    c.execute("INSERT INTO transactions (token_address, tx_hash, action, price, quantity, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                              (token_address, tx_hash, 'sell', current_price, amount / current_price, datetime.now().isoformat()))
                await app.log(f"LIMIT {action.upper()} {amount} SOL of {token_address} at {price_target} - Tx: {tx_hash}")
                del LIMIT_ORDERS[token_address]
                conn.commit()
                await app.sync_holdings()
            elif datetime.now().timestamp() > expiry:
                del LIMIT_ORDERS[token_address]
                await app.log(f"Limit order for {token_address} expired!")
    conn.close()

def blacklist_entity(type, address, reason):
    conn = sqlite3.connect('solana_data.db')
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO blacklists (type, address, reason) VALUES (?, ?, ?)", (type, address, reason))
    conn.commit()
    conn.close()

# Main execution
if __name__ == "__main__":
    init_db()
    app = SolanaTraderApp()
    app.run()
