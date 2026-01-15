// Vue应用主逻辑
const { createApp } = Vue;

const API_BASE_URL = 'http://localhost:8000';

createApp({
    data() {
        return {
            currentView: 'recommend',
            loading: false,
            
            // 推荐表单
            recommendForm: {
                user_id: 0,
                top_k: 10,
                rec_type: 'poi'
            },
            recommendations: [],
            
            // 搜索表单
            searchForm: {
                query: '',
                location: '',
                sort_by: 'rating'
            },
            searchResults: [],
            searchIntent: null,
            
            // 运营分析
            operationForm: {
                poi_id: 0
            },
            poiDetail: null,
            operationAdvice: null,
            qaQuestion: '',
            qaAnswer: null,
            
            // 统计数据
            stats: {
                users: 0,
                pois: 0,
                spus: 0,
                orders: 0
            }
        };
    },
    
    mounted() {
        // 页面加载时获取统计数据
        this.loadStats();
    },
    
    methods: {
        // 获取推荐
        async getRecommendations() {
            if (!this.recommendForm.user_id) {
                alert('请输入用户ID');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${API_BASE_URL}/api/recommend`, this.recommendForm);
                this.recommendations = response.data.recommendations;
                console.log('推荐结果:', response.data);
            } catch (error) {
                console.error('获取推荐失败:', error);
                alert('获取推荐失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 搜索商家
        async searchPois() {
            if (!this.searchForm.query && !this.searchForm.location) {
                alert('请输入搜索条件');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${API_BASE_URL}/api/search`, this.searchForm);
                this.searchResults = response.data.results;
                this.searchIntent = response.data.intent;
                console.log('搜索结果:', response.data);
            } catch (error) {
                console.error('搜索失败:', error);
                alert('搜索失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 查看商家详情
        async viewPoiDetail(poiId) {
            this.currentView = 'operation';
            this.operationForm.poi_id = poiId;
            await this.getPoiInfo(poiId);
        },
        
        // 获取商家信息
        async getPoiInfo(poiId) {
            this.loading = true;
            try {
                const response = await axios.get(`${API_BASE_URL}/api/poi/${poiId}`);
                this.poiDetail = {
                    ...response.data.info,
                    ...response.data.stats
                };
                console.log('商家详情:', response.data);
            } catch (error) {
                console.error('获取商家详情失败:', error);
                alert('获取商家详情失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 获取运营分析
        async getOperationAnalysis() {
            if (!this.operationForm.poi_id) {
                alert('请输入商家ID');
                return;
            }
            
            // 先获取商家信息
            await this.getPoiInfo(this.operationForm.poi_id);
            
            // 再获取AI分析
            this.loading = true;
            try {
                const response = await axios.post(`${API_BASE_URL}/api/operation/analysis`, 
                    this.operationForm);
                this.operationAdvice = response.data.analysis;
                console.log('运营分析:', response.data);
            } catch (error) {
                console.error('获取运营分析失败:', error);
                alert('获取运营分析失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 智能问答
        async askQuestion() {
            if (!this.qaQuestion) {
                alert('请输入问题');
                return;
            }
            
            if (!this.operationForm.poi_id) {
                alert('请先选择一个商家');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${API_BASE_URL}/api/qa/answer`, {
                    poi_id: this.operationForm.poi_id,
                    question: this.qaQuestion
                });
                this.qaAnswer = response.data.answer;
                console.log('问答结果:', response.data);
            } catch (error) {
                console.error('问答失败:', error);
                alert('问答失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 加载统计数据
        async loadStats() {
            try {
                const response = await axios.get(`${API_BASE_URL}/api/stats`);
                this.stats = response.data;
                console.log('统计数据:', response.data);
            } catch (error) {
                console.error('加载统计数据失败:', error);
            }
        }
    },
    
    watch: {
        // 切换视图时的处理
        currentView(newView) {
            console.log('切换到视图:', newView);
            
            // 切换到统计视图时刷新数据
            if (newView === 'stats') {
                this.loadStats();
            }
        }
    }
}).mount('#app');
