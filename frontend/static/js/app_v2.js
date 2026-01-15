/**
 * Vue应用 V2 - 集成第三阶段所有功能
 */

const { createApp } = Vue;

createApp({
    data() {
        return {
            currentView: 'recommend',
            loading: false,
            
            // 推荐表单
            recommendForm: {
                user_id: '',
                top_k: 10,
                use_llm: false
            },
            recommendations: [],
            
            // RAG搜索
            ragForm: {
                query: '',
                top_k: 20
            },
            ragResult: null,
            
            // 运营分析
            operationForm: {
                poi_id: ''
            },
            operationReport: null,
            
            // 评论摘要
            commentForm: {
                poi_id: ''
            },
            commentSummary: null,
            
            // 智能问答
            qaForm: {
                question: '',
                context: null
            },
            qaHistory: [],
            
            // 统计数据
            stats: null,
            
            // API配置
            apiBase: 'http://localhost:8000'
        };
    },
    
    methods: {
        // 智能推荐
        async getRecommendations() {
            if (!this.recommendForm.user_id) {
                alert('请输入用户ID');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${this.apiBase}/api/recommend`, this.recommendForm);
                this.recommendations = response.data.recommendations || [];
                
                if (this.recommendations.length === 0) {
                    alert('未找到推荐结果');
                }
            } catch (error) {
                console.error('推荐失败:', error);
                alert('推荐失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // RAG智能搜索
        async ragSearch() {
            if (!this.ragForm.query.trim()) {
                alert('请输入搜索内容');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${this.apiBase}/api/rag/search`, this.ragForm);
                this.ragResult = response.data;
                
                if (!this.ragResult.results || this.ragResult.results.length === 0) {
                    alert('未找到相关结果');
                }
            } catch (error) {
                console.error('RAG搜索失败:', error);
                alert('搜索失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // Agent运营分析
        async getOperationAnalysis() {
            if (!this.operationForm.poi_id) {
                alert('请输入商家ID');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${this.apiBase}/api/operation/analysis`, this.operationForm);
                this.operationReport = response.data.report;
                
                if (!this.operationReport) {
                    alert('分析失败');
                }
            } catch (error) {
                console.error('运营分析失败:', error);
                alert('分析失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 评论摘要
        async getCommentSummary() {
            if (!this.commentForm.poi_id) {
                alert('请输入商家ID');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${this.apiBase}/api/comment/summary`, this.commentForm);
                this.commentSummary = response.data;
                
                if (this.commentSummary.error) {
                    alert('摘要生成失败: ' + this.commentSummary.error);
                }
            } catch (error) {
                console.error('评论摘要失败:', error);
                alert('摘要失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 智能问答
        async askQuestion() {
            if (!this.qaForm.question.trim()) {
                alert('请输入问题');
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${this.apiBase}/api/qa/answer`, this.qaForm);
                
                this.qaHistory.unshift({
                    question: this.qaForm.question,
                    answer: response.data.answer
                });
                
                // 清空输入
                this.qaForm.question = '';
                
                // 最多保留10条历史
                if (this.qaHistory.length > 10) {
                    this.qaHistory = this.qaHistory.slice(0, 10);
                }
            } catch (error) {
                console.error('问答失败:', error);
                alert('问答失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 加载统计数据
        async loadStats() {
            this.loading = true;
            try {
                const response = await axios.get(`${this.apiBase}/api/stats`);
                this.stats = response.data;
            } catch (error) {
                console.error('加载统计失败:', error);
                alert('加载失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        },
        
        // 构建向量索引
        async buildVectorIndex() {
            if (!confirm('构建向量索引可能需要几分钟时间，确认继续？')) {
                return;
            }
            
            this.loading = true;
            try {
                const response = await axios.post(`${this.apiBase}/api/vector/build`);
                alert('向量索引构建成功！\n' + JSON.stringify(response.data.stats, null, 2));
                await this.loadStats(); // 刷新统计
            } catch (error) {
                console.error('构建索引失败:', error);
                alert('构建失败: ' + (error.response?.data?.detail || error.message));
            } finally {
                this.loading = false;
            }
        }
    },
    
    mounted() {
        // 页面加载时获取统计数据
        this.loadStats();
        
        // 检查API连接
        axios.get(`${this.apiBase}/health`)
            .then(response => {
                console.log('API状态:', response.data);
            })
            .catch(error => {
                console.error('API连接失败:', error);
                alert('无法连接到后端服务，请确保已启动backend/app_v2.py');
            });
    }
}).mount('#app');
