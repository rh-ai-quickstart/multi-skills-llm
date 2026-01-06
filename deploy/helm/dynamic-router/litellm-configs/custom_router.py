from litellm.integrations.custom_logger import CustomLogger
import litellm
from litellm.proxy.proxy_server import UserAPIKeyAuth, DualCache
from typing import Optional, Literal
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import FastEmbedEncoder
import json
import re


# Define routes for complexity-based routing
simple_tasks = Route(
    name="simple-xeon",
    utterances=[
        "What is",
        "Define",
        "Explain briefly",
        "Simple question",
        "Quick answer",
        "Tell me about",
        "What does",
        "How do you spell",
        "What's the meaning of",
        "Give me a summary",
        "What time",
        "What date",
        "Basic information",
        "Simple definition",
        "Short explanation",
        "Who is",
        "Where is",
        "When did",
        "How many",
        "List the",
        "Name the",
        "What color",
        "Yes or no",
        "True or false",
        "Quick question",
    ],
)

complex_tasks = Route(
    name="complex-gaudi",
    utterances=[
        "Analyze the complex relationship",
        "Write a detailed report",
        "Provide a comprehensive analysis",
        "Create a detailed plan",
        "Develop a strategy",
        "Write code for",
        "Debug this complex problem",
        "Solve this mathematical equation",
        "Generate a detailed explanation",
        "Compare and contrast multiple",
        "Perform deep analysis",
        "Research and provide insights",
        "Create a detailed algorithm",
        "Design a complex system",
        "Provide step-by-step solution",
        "Explain in depth",
        "Write an essay about",
        "Synthesize information from",
        "Evaluate the pros and cons",
        "Build a machine learning model",
        "Optimize this code",
        "Architect a solution",
        "Troubleshoot this complex issue",
        "Create a comprehensive guide",
        "Develop a business plan",
    ],
)

# Define Xeon database routes based on query type
billing_payments_database = Route(
    name="billing-payments",
    utterances=[
        "charged",
        "payment failed",
        "price discrepancy",
        "invoice",
        "refund",
        "price",
        "billing issue",
        "payment method",
        "credit card",
        "debit card",
        "subscription cost",
        "monthly fee",
        "annual fee",
        "overcharged",
        "double charged",
        "payment not processed",
        "update payment",
        "cancel subscription",
        "renew subscription",
        "payment history",
        "receipt",
        "transaction",
        "discount code",
        "coupon",
        "promo code",
        "billing cycle",
        "due date",
        "outstanding balance",
        "payment confirmation",
        "autopay",
    ],
)

product_qna_database = Route(
    name="product-qna",
    utterances=[
        "specs",
        "size",
        "dimensions",
        "compatible",
        "colors",
        "does it support",
        "product features",
        "how does it work",
        "what materials",
        "weight",
        "capacity",
        "battery life",
        "warranty information",
        "product manual",
        "user guide",
        "available sizes",
        "color options",
        "model comparison",
        "product specifications",
        "technical details",
        "does it fit",
        "works with",
        "compatible with",
        "system requirements",
        "product availability",
        "in stock",
        "product quality",
        "durability",
        "how to use",
        "assembly instructions",
    ],
)

shipping_return_database = Route(
    name="shipping-returns",
    utterances=[
        "tracking",
        "lost package",
        "delivery",
        "return label",
        "shipping status",
        "where is my order",
        "estimated delivery",
        "shipping cost",
        "free shipping",
        "expedited shipping",
        "international shipping",
        "return policy",
        "how to return",
        "exchange item",
        "refund for return",
        "package not arrived",
        "damaged package",
        "wrong item received",
        "missing item",
        "delivery address",
        "change shipping address",
        "shipping options",
        "carrier information",
        "fedex tracking",
        "ups tracking",
        "usps tracking",
        "delivery time",
        "order status",
        "package delayed",
        "return window",
    ],
)

account_info_database = Route(
    name="account-info",
    utterances=[
        "reset password",
        "change email",
        "update profile",
        "account settings",
        "login issue",
        "forgot password",
        "username",
        "sign up",
        "create account",
        "delete account",
        "deactivate account",
        "account verification",
        "two-factor authentication",
        "security settings",
        "notification preferences",
        "email preferences",
        "change phone number",
        "update address",
        "account history",
        "order history",
        "saved payment methods",
        "wishlist",
        "favorites",
        "account locked",
        "unlock account",
        "privacy settings",
        "data export",
        "account information",
        "membership status",
        "loyalty points",
    ],
)

# Define Gaudi database routes based on query type
setup_info_database = Route(
    name="setup-info",
    utterances=[
        "how to install",
        "installation guide",
        "setup instructions",
        "getting started",
        "initial configuration",
        "first time setup",
        "connect device",
        "pair device",
        "bluetooth pairing",
        "wifi setup",
        "network configuration",
        "driver installation",
        "software installation",
        "download app",
        "activate device",
        "register product",
        "unboxing",
        "quick start",
        "setup wizard",
        "configure settings",
        "system setup",
        "hardware setup",
        "mount instructions",
        "assembly guide",
        "connection guide",
    ],
)

reported_issues_database = Route(
    name="reported-issues",
    utterances=[
        "not working",
        "broken",
        "error message",
        "bug report",
        "known issues",
        "common problems",
        "troubleshooting",
        "device malfunction",
        "app crash",
        "freezing",
        "slow performance",
        "connectivity issues",
        "won't turn on",
        "battery drain",
        "overheating",
        "display issues",
        "audio problems",
        "sync issues",
        "update failed",
        "software bug",
        "hardware defect",
        "intermittent problem",
        "random shutdown",
        "unresponsive",
        "error code",
    ],
)

class ComplexityBasedRouter(CustomLogger):
    def __init__(self):
        # Removed medical_tasks from routes - only simple and complex now
        self.routes = [simple_tasks, complex_tasks]
        self.encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5", score_threshold=0.5)
        self.routelayer = RouteLayer(encoder=self.encoder, routes=self.routes)
        
        # Complexity indicators - these patterns suggest complex tasks
        self.complexity_patterns = [
            r'\b(analyze|analysis|detailed|comprehensive|complex|algorithm|debug|strategy)\b',
            r'\b(compare.*contrast|step.*by.*step|in.*depth|thoroughly)\b',
            r'\b(code|programming|function|class|method|variable)\b',
            r'\b(mathematical|equation|formula|calculation|solve)\b',
            r'\b(research|investigate|examine|explore)\b',
            r'\b(design|architect|implement|develop)\b'
        ]
        
        # Simple task indicators
        self.simple_patterns = [
            r'^\s*(what\s+is|define|explain\s+briefly|tell\s+me\s+about)\b',
            r'^\s*(how\s+do\s+you\s+spell|what.*meaning|quick\s+answer)\b',
            r'^\s*(what\s+time|what\s+date|basic\s+information)\b'
        ]

    def assess_complexity(self, message: str) -> str:
        """
        Assess the complexity of a message using multiple factors:
        1. Length (longer messages often indicate complexity)
        2. Pattern matching for complex/simple keywords
        3. Semantic routing results
        4. Question structure
        """
        message_lower = message.lower()
        
        # Factor 1: Length-based assessment
        word_count = len(message.split())
        length_score = min(word_count / 20.0, 1.0)  # Normalize to 0-1, cap at 20 words
        
        # Factor 2: Pattern matching
        complexity_score = 0.0
        simple_score = 0.0
        
        for pattern in self.complexity_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                complexity_score += 0.3
                
        for pattern in self.simple_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                simple_score += 0.4
        
        # Factor 3: Semantic routing
        route = self.routelayer(message)
        route_metrics = self.routelayer.retrieve_multiple_routes(message)
        
        semantic_score = 0.0
        if route_metrics:
            for route_choice in route_metrics:
                if "complex" in route_choice.name:
                    semantic_score += route_choice.similarity_score * 0.5
                elif "simple" in route_choice.name:
                    semantic_score -= route_choice.similarity_score * 0.3
        
        # Combine all factors
        total_complexity = length_score + complexity_score + semantic_score - simple_score
        
        print(f"Complexity Assessment: length={length_score:.2f}, patterns={complexity_score:.2f}, "
              f"simple_patterns={simple_score:.2f}, semantic={semantic_score:.2f}, "
              f"total={total_complexity:.2f}")
        
        # Decision threshold
        if total_complexity > 0.6:
            return "complex-gaudi"
        else:
            return "simple-xeon"

    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: Literal[
            "completion",
            "text_completion", 
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ]): 
        msg = data['messages'][-1]['content']
        
        # Assess complexity and route accordingly
        selected_model = self.assess_complexity(msg)
        
        # Try semantic routing first
        route = self.routelayer(msg)
        route_metrics = self.routelayer.retrieve_multiple_routes(msg)
        
        print(f"Semantic route metrics: {route_metrics}")
        
        if route_metrics and route_metrics[0].similarity_score > 0.7:
            # High confidence semantic match
            selected_model = route.name
            print(f"High confidence semantic routing to: {selected_model}")
        else:
            # Use complexity-based routing
            print(f"Complexity-based routing to: {selected_model}")
        
        data["model"] = selected_model
        #data["selected_model"] = selected_model  # <-- Add this line
        data.setdefault("_router_meta", {})["final_model"] = selected_model
        print(f"Final model selection: {selected_model}")           
        return data 

    async def async_post_call_failure_hook(
        self, 
        request_data: dict,
        original_exception: Exception, 
        user_api_key_dict: UserAPIKeyAuth
    ):
        print(f"Request failed: {original_exception}")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        pass

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation", "moderation", "audio_transcription"],
    ):
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        try:
            text = response.decode("utf-8", "ignore") if isinstance(response, (bytes, bytearray)) else str(response)
            # (Optional) LOG ONLY – do not modify stream
            # for line in text.splitlines():
            #     if line.startswith("data:"):
            #         payload = line[5:].strip()
            #         if payload and payload != "[DONE]":
            #             try:
            #                 obj = json.loads(payload)
            #                 if "model" in obj: print(f"[stream] model={obj['model']}")
            #             except Exception:
            #                 pass
        except Exception:
        # swallow errors to avoid breaking post_success
            pass
    # IMPORTANT: don't return anything; hook is side-effect only
        return

class TypeBasedSimpleRouter(CustomLogger):
    def __init__(self):
        self.routes = [billing_payments_database, product_qna_database, shipping_return_database, account_info_database]
        self.encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5", score_threshold=0.5)
        self.routelayer = RouteLayer(encoder=self.encoder, routes=self.routes)
        
        # Billing and payments patterns
        self.billing_patterns = [
            r'\b(bill|billing|invoice|payment|charge|charged|price|cost|fee|refund)\b',
            r'\b(subscription|renew|cancel.*subscription|upgrade|downgrade)\b',
            r'\b(credit card|debit|transaction|receipt|discount|coupon|promo)\b',
        ]
        
        # Product Q&A patterns
        self.product_patterns = [
            r'\b(product|item|specs|specifications|features|dimensions|size|color)\b',
            r'\b(compatible|compatibility|support|work with|fit)\b',
            r'\b(warranty|guarantee|material|weight|capacity)\b',
        ]
        
        # Shipping and returns patterns
        self.shipping_patterns = [
            r'\b(ship|shipping|delivery|deliver|track|tracking|package)\b',
            r'\b(return|exchange|lost|missing|damaged|arrive|arrived)\b',
            r'\b(address|carrier|fedex|ups|usps|estimated|eta)\b',
        ]
        
        # Account info patterns
        self.account_patterns = [
            r'\b(account|profile|login|password|username|email|phone)\b',
            r'\b(settings|preferences|notification|update.*info|change.*info)\b',
            r'\b(sign up|register|delete account|deactivate|verify)\b',
        ]

    def assess_query_type(self, message: str) -> str:
        """
        Assess the query type based on pattern matching and semantic routing.
        Routes to: billing-payments, product-qna, shipping-returns, or account-info
        """
        message_lower = message.lower()
        
        # Score each category based on pattern matching
        scores = {
            "billing-payments": 0.0,
            "product-qna": 0.0,
            "shipping-returns": 0.0,
            "account-info": 0.0,
        }
        
        for pattern in self.billing_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                scores["billing-payments"] += 0.3
                
        for pattern in self.product_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                scores["product-qna"] += 0.3
                
        for pattern in self.shipping_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                scores["shipping-returns"] += 0.3
                
        for pattern in self.account_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                scores["account-info"] += 0.3
        
        # Get the highest scoring category from pattern matching
        max_pattern_score = max(scores.values())
        pattern_result = max(scores, key=scores.get) if max_pattern_score > 0 else None
        
        print(f"Pattern scores: {scores}")
        
        # Use semantic routing for additional confidence
        route = self.routelayer(message)
        route_metrics = self.routelayer.retrieve_multiple_routes(message)
        
        if route_metrics:
            print(f"Semantic route metrics: {route_metrics}")
            # Boost scores based on semantic routing
            for route_choice in route_metrics:
                if route_choice.name in scores:
                    scores[route_choice.name] += route_choice.similarity_score * 0.5
        
        # Determine final route
        final_scores = scores
        best_route = max(final_scores, key=final_scores.get)
        
        print(f"Final scores: {final_scores}, Selected: {best_route}")
        
        # Default to product-qna if no clear match
        if final_scores[best_route] < 0.2:
            best_route = "product-qna"
            print(f"Low confidence, defaulting to: {best_route}")
        
        return best_route

    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: Literal[
            "completion",
            "text_completion", 
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ]): 
        msg = data['messages'][-1]['content']
        
        # Assess query type and route accordingly
        selected_model = self.assess_query_type(msg)
        
        # Try semantic routing for high confidence matches
        route = self.routelayer(msg)
        route_metrics = self.routelayer.retrieve_multiple_routes(msg)
        
        print(f"Semantic route metrics: {route_metrics}")
        
        if route_metrics and route_metrics[0].similarity_score > 0.7:
            # High confidence semantic match
            selected_model = route.name
            print(f"High confidence semantic routing to: {selected_model}")
        else:
            # Use type-based routing
            print(f"Type-based routing to: {selected_model}")
        
        data["model"] = selected_model
        data.setdefault("_router_meta", {})["final_model"] = selected_model
        print(f"Final model selection: {selected_model}")           
        return data 

    async def async_post_call_failure_hook(
        self, 
        request_data: dict,
        original_exception: Exception, 
        user_api_key_dict: UserAPIKeyAuth
    ):
        print(f"Request failed: {original_exception}")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        pass

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation", "moderation", "audio_transcription"],
    ):
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        pass

class TypeBasedComplexRouter(CustomLogger):
    def __init__(self):
        self.routes = [setup_info_database, reported_issues_database]
        self.encoder = FastEmbedEncoder(name="BAAI/bge-small-en-v1.5", score_threshold=0.5)
        self.routelayer = RouteLayer(encoder=self.encoder, routes=self.routes)
        
        # Setup and installation patterns
        self.setup_patterns = [
            r'\b(install|installation|setup|set up|configure|configuration)\b',
            r'\b(getting started|first time|initial|activate|register)\b',
            r'\b(connect|pair|pairing|bluetooth|wifi|network)\b',
            r'\b(driver|software|download|app|unboxing|quick start)\b',
        ]
        
        # Reported issues and troubleshooting patterns
        self.issues_patterns = [
            r'\b(not working|broken|error|bug|issue|problem)\b',
            r'\b(crash|freezing|slow|malfunction|defect)\b',
            r'\b(won\'t|can\'t|unable|failed|failing)\b',
            r'\b(troubleshoot|fix|resolve|repair|help)\b',
        ]

    def assess_query_type(self, message: str) -> str:
        """
        Assess the query type based on pattern matching and semantic routing.
        Routes to: setup-info or reported-issues
        """
        message_lower = message.lower()
        
        # Score each category based on pattern matching
        scores = {
            "setup-info": 0.0,
            "reported-issues": 0.0,
        }
        
        for pattern in self.setup_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                scores["setup-info"] += 0.3
                
        for pattern in self.issues_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                scores["reported-issues"] += 0.3
        
        print(f"Pattern scores: {scores}")
        
        # Use semantic routing for additional confidence
        route = self.routelayer(message)
        route_metrics = self.routelayer.retrieve_multiple_routes(message)
        
        if route_metrics:
            print(f"Semantic route metrics: {route_metrics}")
            # Boost scores based on semantic routing
            for route_choice in route_metrics:
                if route_choice.name in scores:
                    scores[route_choice.name] += route_choice.similarity_score * 0.5
        
        # Determine final route
        best_route = max(scores, key=scores.get)
        
        print(f"Final scores: {scores}, Selected: {best_route}")
        
        # Default to setup-info if no clear match
        if scores[best_route] < 0.2:
            best_route = "setup-info"
            print(f"Low confidence, defaulting to: {best_route}")
        
        return best_route

    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: Literal[
            "completion",
            "text_completion", 
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ]): 
        msg = data['messages'][-1]['content']
        
        # Assess query type and route accordingly
        selected_model = self.assess_query_type(msg)
        
        # Try semantic routing for high confidence matches
        route = self.routelayer(msg)
        route_metrics = self.routelayer.retrieve_multiple_routes(msg)
        
        print(f"Semantic route metrics: {route_metrics}")
        
        if route_metrics and route_metrics[0].similarity_score > 0.7:
            # High confidence semantic match
            selected_model = route.name
            print(f"High confidence semantic routing to: {selected_model}")
        else:
            # Use type-based routing
            print(f"Type-based routing to: {selected_model}")
        
        data["model"] = selected_model
        data.setdefault("_router_meta", {})["final_model"] = selected_model
        print(f"Final model selection: {selected_model}")           
        return data 

    async def async_post_call_failure_hook(
        self, 
        request_data: dict,
        original_exception: Exception, 
        user_api_key_dict: UserAPIKeyAuth
    ):
        print(f"Request failed: {original_exception}")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        pass

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation", "moderation", "audio_transcription"],
    ):
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        pass

class ChainedRouter(CustomLogger):
    """
    A chained router that first assesses query complexity, then routes to
    the appropriate type-based router:
    - simple-xeon -> TypeBasedSimpleRouter (billing, product, shipping, account)
    - complex-gaudi -> TypeBasedComplexRouter (setup, reported issues)
    """
    def __init__(self):
        self.complexity_router = ComplexityBasedRouter()
        self.simple_type_router = TypeBasedSimpleRouter()
        self.complex_type_router = TypeBasedComplexRouter()

    def route_message(self, message: str) -> str:
        """
        Chain the routers together:
        1. First assess complexity (simple-xeon vs complex-gaudi)
        2. Then route to appropriate type-based router
        """
        # Step 1: Assess complexity
        complexity_result = self.complexity_router.assess_complexity(message)
        print(f"[ChainedRouter] Complexity assessment: {complexity_result}")
        
        # Step 2: Route to appropriate type-based router
        if complexity_result == "simple-xeon":
            final_route = self.simple_type_router.assess_query_type(message)
            print(f"[ChainedRouter] Simple path -> TypeBasedSimpleRouter -> {final_route}")
        else:  # complex-gaudi
            final_route = self.complex_type_router.assess_query_type(message)
            print(f"[ChainedRouter] Complex path -> TypeBasedComplexRouter -> {final_route}")
        
        return final_route

    async def async_pre_call_hook(self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: Literal[
            "completion",
            "text_completion", 
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ]): 
        msg = data['messages'][-1]['content']
        
        # Use chained routing logic
        selected_model = self.route_message(msg)
        
        data["model"] = selected_model
        data.setdefault("_router_meta", {})["final_model"] = selected_model
        print(f"[ChainedRouter] Final model selection: {selected_model}")
        return data 

    async def async_post_call_failure_hook(
        self, 
        request_data: dict,
        original_exception: Exception, 
        user_api_key_dict: UserAPIKeyAuth
    ):
        print(f"[ChainedRouter] Request failed: {original_exception}")

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        pass

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation", "moderation", "audio_transcription"],
    ):
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ):
        pass


proxy_handler_instance = ComplexityBasedRouter()
proxy_handler_instance_2 = TypeBasedSimpleRouter()
proxy_handler_instance_3 = TypeBasedComplexRouter()
proxy_handler_instance_chained = ChainedRouter()
