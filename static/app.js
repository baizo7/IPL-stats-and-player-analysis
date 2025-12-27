const { createApp } = Vue;

createApp({
    delimiters: ['[[', ']]'],  // Use custom delimiters to avoid Jinja2 conflict
    data() {
        return {
            teams: [],
            team1: '',
            team2: '',
            currentView: 'phase',
            loading: false,
            team1PhaseData: [],
            team2PhaseData: [],
            team1MatchupData: [],
            team2MatchupData: [],
            bowlerTypes: [],
            team1Summary: {},
            team2Summary: {},
            team1TopBatters: [],
            team2TopBatters: [],
            team1OverProgression: [],
            team2OverProgression: [],
            runRateChart: null,
            boundariesChart: null,
            overChart1: null,
            overChart2: null
        };
    },
    async mounted() {
        // GSAP Animations
        this.initAnimations();
        
        // Load teams
        await this.loadTeams();
        
        // Set default teams
        if (this.teams.length >= 2) {
            this.team1 = this.teams[0];
            this.team2 = this.teams[1];
            await this.loadData();
        }
    },
    methods: {
        initAnimations() {
            gsap.registerPlugin(ScrollTrigger);
            
            // Header animation
            gsap.from('.header', {
                y: -100,
                opacity: 0,
                duration: 1,
                ease: 'power3.out'
            });
            
            // Team selector animation
            gsap.from('.team-select-box', {
                scale: 0,
                opacity: 0,
                duration: 0.8,
                stagger: 0.2,
                delay: 0.5,
                ease: 'back.out(1.7)'
            });
            
            // VS divider animation
            gsap.from('.vs-divider', {
                scale: 0,
                rotation: 360,
                duration: 1,
                delay: 0.7,
                ease: 'elastic.out(1, 0.5)'
            });
        },
        
        animateCards() {
            // Animate phase cards
            gsap.from('.phase-card', {
                x: -100,
                opacity: 0,
                duration: 0.6,
                stagger: 0.1,
                ease: 'power2.out'
            });
            
            // Animate summary stats
            gsap.from('.summary-stat', {
                scale: 0,
                opacity: 0,
                duration: 0.5,
                stagger: 0.1,
                ease: 'back.out(1.7)'
            });
            
            // Animate charts
            gsap.from('.chart-container', {
                y: 50,
                opacity: 0,
                duration: 0.8,
                stagger: 0.2,
                ease: 'power3.out'
            });
        },
        
        async loadTeams() {
            try {
                const response = await fetch('/api/teams');
                this.teams = await response.json();
            } catch (error) {
                console.error('Error loading teams:', error);
            }
        },
        
        async loadData() {
            if (!this.team1 || !this.team2) return;
            
            this.loading = true;
            
            try {
                await Promise.all([
                    this.loadPhaseData(),
                    this.loadMatchupData(),
                    this.loadSummaryData(),
                    this.loadOverProgression()
                ]);
                
                // Animate after data loads
                setTimeout(() => this.animateCards(), 100);
                
            } catch (error) {
                console.error('Error loading data:', error);
            } finally {
                this.loading = false;
            }
        },
        
        async loadPhaseData() {
            const [response1, response2] = await Promise.all([
                fetch(`/api/phase-analysis?team=${encodeURIComponent(this.team1)}`),
                fetch(`/api/phase-analysis?team=${encodeURIComponent(this.team2)}`)
            ]);
            
            this.team1PhaseData = await response1.json();
            this.team2PhaseData = await response2.json();
            
            this.updateRunRateChart();
            this.updateBoundariesChart();
        },
        
        async loadMatchupData() {
            const [response1, response2] = await Promise.all([
                fetch(`/api/matchup-analysis?team=${encodeURIComponent(this.team1)}`),
                fetch(`/api/matchup-analysis?team=${encodeURIComponent(this.team2)}`)
            ]);
            
            const data1 = await response1.json();
            const data2 = await response2.json();
            
            this.team1MatchupData = data1.data || [];
            this.team2MatchupData = data2.data || [];
            this.bowlerTypes = data1.bowler_types || [];
        },
        
        async loadSummaryData() {
            const [response1, response2, batters1, batters2] = await Promise.all([
                fetch(`/api/team-summary?team=${encodeURIComponent(this.team1)}`),
                fetch(`/api/team-summary?team=${encodeURIComponent(this.team2)}`),
                fetch(`/api/top-batters?team=${encodeURIComponent(this.team1)}&n=5`),
                fetch(`/api/top-batters?team=${encodeURIComponent(this.team2)}&n=5`)
            ]);
            
            this.team1Summary = await response1.json();
            this.team2Summary = await response2.json();
            this.team1TopBatters = await batters1.json();
            this.team2TopBatters = await batters2.json();
        },
        
        async loadOverProgression() {
            const [response1, response2] = await Promise.all([
                fetch(`/api/over-progression?team=${encodeURIComponent(this.team1)}`),
                fetch(`/api/over-progression?team=${encodeURIComponent(this.team2)}`)
            ]);
            
            this.team1OverProgression = await response1.json();
            this.team2OverProgression = await response2.json();
            
            this.updateOverProgressionCharts();
        },
        
        updateRunRateChart() {
            const ctx = document.getElementById('runRateChart');
            if (!ctx) return;
            
            if (this.runRateChart) {
                this.runRateChart.destroy();
            }
            
            this.runRateChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: this.team1PhaseData.map(p => p.phase),
                    datasets: [
                        {
                            label: this.team1,
                            data: this.team1PhaseData.map(p => p.run_rate),
                            backgroundColor: 'rgba(102, 126, 234, 0.8)',
                            borderColor: 'rgba(102, 126, 234, 1)',
                            borderWidth: 2
                        },
                        {
                            label: this.team2,
                            data: this.team2PhaseData.map(p => p.run_rate),
                            backgroundColor: 'rgba(118, 75, 162, 0.8)',
                            borderColor: 'rgba(118, 75, 162, 1)',
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Run Rate by Phase',
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Run Rate'
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeInOutQuart'
                    }
                }
            });
        },
        
        updateBoundariesChart() {
            const ctx = document.getElementById('boundariesChart');
            if (!ctx) return;
            
            if (this.boundariesChart) {
                this.boundariesChart.destroy();
            }
            
            this.boundariesChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: this.team1PhaseData.map(p => p.phase),
                    datasets: [
                        {
                            label: this.team1,
                            data: this.team1PhaseData.map(p => p.boundaries),
                            borderColor: 'rgba(102, 126, 234, 1)',
                            backgroundColor: 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: this.team2,
                            data: this.team2PhaseData.map(p => p.boundaries),
                            borderColor: 'rgba(118, 75, 162, 1)',
                            backgroundColor: 'rgba(118, 75, 162, 0.1)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Boundaries by Phase',
                            font: { size: 16, weight: 'bold' }
                        },
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Boundaries'
                            }
                        }
                    },
                    animation: {
                        duration: 1500,
                        easing: 'easeInOutQuart'
                    }
                }
            });
        },
        
        updateOverProgressionCharts() {
            // Team 1 Chart
            const ctx1 = document.getElementById('overProgressionChart1');
            if (ctx1) {
                if (this.overChart1) {
                    this.overChart1.destroy();
                }
                
                this.overChart1 = new Chart(ctx1, {
                    type: 'line',
                    data: {
                        labels: this.team1OverProgression.map(o => `Over ${o.over}`),
                        datasets: [{
                            label: 'Cumulative Runs',
                            data: this.team1OverProgression.map(o => o.cumulative_runs),
                            borderColor: 'rgba(102, 126, 234, 1)',
                            backgroundColor: 'rgba(102, 126, 234, 0.2)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.3
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Runs'
                                }
                            }
                        },
                        animation: {
                            duration: 2000,
                            easing: 'easeInOutQuart'
                        }
                    }
                });
            }
            
            // Team 2 Chart
            const ctx2 = document.getElementById('overProgressionChart2');
            if (ctx2) {
                if (this.overChart2) {
                    this.overChart2.destroy();
                }
                
                this.overChart2 = new Chart(ctx2, {
                    type: 'line',
                    data: {
                        labels: this.team2OverProgression.map(o => `Over ${o.over}`),
                        datasets: [{
                            label: 'Cumulative Runs',
                            data: this.team2OverProgression.map(o => o.cumulative_runs),
                            borderColor: 'rgba(118, 75, 162, 1)',
                            backgroundColor: 'rgba(118, 75, 162, 0.2)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.3
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { display: false }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Runs'
                                }
                            }
                        },
                        animation: {
                            duration: 2000,
                            easing: 'easeInOutQuart'
                        }
                    }
                });
            }
        },
        
        getHeatColor(strikeRate) {
            if (strikeRate === 0) return '#cbd5e0';
            if (strikeRate < 80) return '#fc8181';
            if (strikeRate < 120) return '#f6ad55';
            if (strikeRate < 150) return '#68d391';
            return '#48bb78';
        }
    },
    watch: {
        currentView() {
            // Animate view changes
            gsap.from('.content-section', {
                opacity: 0,
                y: 50,
                duration: 0.6,
                ease: 'power2.out'
            });
            
            // Recreate charts when switching views
            this.$nextTick(() => {
                if (this.currentView === 'phase') {
                    this.updateRunRateChart();
                    this.updateBoundariesChart();
                } else if (this.currentView === 'detailed') {
                    this.updateOverProgressionCharts();
                }
            });
        }
    }
}).mount('#app');
