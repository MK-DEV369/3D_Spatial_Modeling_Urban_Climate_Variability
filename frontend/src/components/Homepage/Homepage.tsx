import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import FadeIn from '../reactbits/Animations/FadeIn';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Silk from '../ui/Slik';
import ProfileCard from '../ProfileCard';
import GenericLogo from '../Logos/GenericLogo';
import RVCELogo from '../Logos/RVCELogo';
import VariableProximity from '../VariableProximity';

const Homepage: React.FC = () => {
    const containerRef = useRef(null);
    const navigate = useNavigate();

    const techStack = [
        { name: 'React + TypeScript', category: 'Frontend', icon: '⚛️' },
        { name: 'Three.js', category: '3D Rendering', icon: '🎨' },
        { name: 'Django + PostGIS', category: 'Backend', icon: '🐍' },
        { name: 'GraphCast', category: 'Weather AI', icon: '🌦️' },
        { name: 'ClimaX', category: 'Climate AI', icon: '🌍' },
        { name: 'PostgreSQL', category: 'Database', icon: '🗄️' },
    ];

    const features = [
        {
            title: '3D City Visualization',
            description: 'Interactive 3D maps with OpenStreetMap buildings and satellite overlay',
            icon: '🏙️',
        },
        {
            title: 'AI-Powered Climate Modeling',
            description: 'GraphCast & ClimaX models for weather prediction and climate projection',
            icon: '🤖',
        },
        {
            title: 'Scenario Builder',
            description: 'Remove buildings, add vegetation, and see climate impact in real-time',
            icon: '🌳',
        },
        {
            title: 'Traffic & Pollution Analysis',
            description: 'LSTM-based traffic modeling with AQI and pollution forecasting',
            icon: '🚗',
        },
        {
            title: 'Multi-City Comparison',
            description: 'Compare unplanned cities (Bengaluru) with planned cities (Dubai, Netherlands)',
            icon: '📊',
        },
        {
            title: 'Economic Impact',
            description: 'Calculate economic consequences of poor urban planning',
            icon: '💰',
        },
    ];

    const workflow = [
        { step: 1, title: 'Data Collection', desc: 'OSM buildings, climate, traffic, pollution' },
        { step: 2, title: 'ML Processing', desc: 'GraphCast, ClimaX, LSTM models' },
        { step: 3, title: '3D Visualization', desc: 'Three.js rendering with overlays' },
        { step: 4, title: 'Scenario Analysis', desc: 'Predict climate impact of changes' },
    ];

    const teamMembers = [
        {
            name: 'Member 1',
            usn: '1XX21XXXXX',
            branch: 'Computer Science',
            linkedin: 'https://linkedin.com/in/member1',
            github: 'https://github.com/member1',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member1'
        },
        {
            name: 'Member 2',
            usn: '1XX21XXXXX',
            branch: 'Computer Science',
            linkedin: 'https://linkedin.com/in/member2',
            github: 'https://github.com/member2',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member2'
        },
        {
            name: 'Member 3',
            usn: '1XX21XXXXX',
            branch: 'Computer Science',
            linkedin: 'https://linkedin.com/in/member3',
            github: 'https://github.com/member3',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member3'
        },
        {
            name: 'Member 4',
            usn: '1XX21XXXXX',
            branch: 'Computer Science',
            linkedin: 'https://linkedin.com/in/member4',
            github: 'https://github.com/member4',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member4'
        },
        {
            name: 'Raghavendra CS',
            usn: '1XX21XXXXX',
            branch: 'Computer Science',
            linkedin: 'https://linkedin.com/in/member5',
            github: 'https://github.com/member5',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member5'
        },
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 relative">
            {/* Animated Background */}
            <div className="fixed inset-0 z-0">
                <Silk
                    speed={5}
                    scale={1}
                    color="#7B7481"
                    noiseIntensity={1.5}
                    rotation={0}
                />
            </div>

            {/* Hero Section */}
            <section className="relative min-h-screen flex items-center justify-center px-4 py-20 z-10">
                <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
                <FadeIn delay={0.2} className="text-center z-10 max-w-5xl">
                    <div className="flex items-center justify-center gap-8 mb-8">
                        <GenericLogo className="w-24 h-24 rounded-full overflow-hidden shadow-2xl" />
                        <RVCELogo className="w-24 h-24 overflow-hidden shadow-2xl" />
                    </div>
                    <div ref={containerRef} className="mb-6">
                        <VariableProximity
                            label={'3D Spatial Modeling of Urban Climate Variability'}
                            className={'text-4xl md:text-6xl lg:text-7xl font-bold text-white'}
                            fromFontVariationSettings="'wght' 500, 'opsz' 12"
                            toFontVariationSettings="'wght' 900, 'opsz' 48"
                            containerRef={containerRef}
                            radius={150}
                            falloff='linear'
                            style={{ fontWeight: 'bold' }}
                        />
                    </div>
                    <p className="text-xl md:text-2xl text-blue-200 mb-4">
                        Visualizing the Climate Impact of Unplanned Urban Development
                    </p>
                    <p className="text-lg text-gray-300 mb-8 max-w-3xl mx-auto">
                        A comprehensive AI-powered platform that showcases how unplanned cities like Bengaluru
                        disrupt climate patterns, increase pollution, and impact global warming — compared with
                        planned cities like Dubai and Netherlands.
                    </p>
                    <div className="flex gap-4 justify-center flex-wrap">
                        <Button
                            onClick={() => navigate('/dashboard')}
                            size="lg"
                            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 text-lg font-semibold"
                        >
                            Explore Dashboard
                        </Button>
                        <Button
                            onClick={() => navigate('/viewer3d')}
                            size="lg"
                            className="bg-green-600 hover:bg-green-700 text-white px-8 py-4 text-lg font-semibold"
                        >
                            View 3D Map
                        </Button>
                    </div>
                </FadeIn>
            </section>

            {/* Problem Statement */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.3}>
                        <h2 className="text-4xl font-bold text-white mb-8 text-center">The Problem</h2>
                        <div className="grid md:grid-cols-3 gap-6">
                            <Card className="bg-red-900/30 border-red-500/30">
                                <CardHeader>
                                    <div className="text-4xl mb-2">🌡️</div>
                                    <CardTitle className="text-2xl text-red-300">Heat Island Effect</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-gray-300">
                                        Unplanned building density creates urban heat islands, increasing temperatures by
                                        2-5°C and energy consumption.
                                    </p>
                                </CardContent>
                            </Card>
                            <Card className="bg-orange-900/30 border-orange-500/30">
                                <CardHeader>
                                    <div className="text-4xl mb-2">🚗</div>
                                    <CardTitle className="text-2xl text-orange-300">Traffic Congestion</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-gray-300">
                                        Poor road planning forces longer commutes, increasing pollution and economic
                                        inefficiency.
                                    </p>
                                </CardContent>
                            </Card>
                            <Card className="bg-purple-900/30 border-purple-500/30">
                                <CardHeader>
                                    <div className="text-4xl mb-2">💨</div>
                                    <CardTitle className="text-2xl text-purple-300">Air Pollution</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-gray-300">
                                        Lack of vegetation and poor layout traps pollutants, affecting public health and
                                        climate.
                                    </p>
                                </CardContent>
                            </Card>
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Project Workflow */}
            <section className="px-4 py-20 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.4}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">How It Works</h2>
                        <div className="grid md:grid-cols-4 gap-6">
                            {workflow.map((item, idx) => (
                                <div key={idx} className="relative">
                                    <div className="bg-blue-900/30 p-6 rounded-lg border border-blue-500/30 text-center">
                                        <div className="text-5xl font-bold text-blue-400 mb-3">{item.step}</div>
                                        <h3 className="text-xl font-semibold text-white mb-2">{item.title}</h3>
                                        <p className="text-gray-300 text-sm">{item.desc}</p>
                                    </div>
                                    {idx < workflow.length - 1 && (
                                        <div className="hidden md:block absolute top-1/2 -right-3 transform -translate-y-1/2 text-blue-400 text-2xl">
                                            →
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Features */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.5}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Key Features</h2>
                        <div className="grid md:grid-cols-3 gap-6">
                            {features.map((feature, idx) => (
                                <Card key={idx} className="bg-slate-900/50 border-slate-700 hover:border-slate-500 transition-colors">
                                    <CardHeader>
                                        <div className="text-5xl mb-2">{feature.icon}</div>
                                        <CardTitle className="text-xl">{feature.title}</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-gray-300">{feature.description}</p>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Technology Stack */}
            <section className="px-4 py-20 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.6}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Technology Stack</h2>
                        <div className="grid md:grid-cols-3 gap-6">
                            {techStack.map((tech, idx) => (
                                <div
                                    key={idx}
                                    className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 p-6 rounded-lg border border-blue-500/30 text-center"
                                >
                                    <div className="text-4xl mb-3">{tech.icon}</div>
                                    <h3 className="text-xl font-semibold text-white mb-1">{tech.name}</h3>
                                    <p className="text-sm text-blue-300">{tech.category}</p>
                                </div>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Use Cases */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.7}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">
                            What This Platform Can Be Used For
                        </h2>
                        <div className="grid md:grid-cols-2 gap-6">
                            <div className="bg-green-900/20 p-6 rounded-lg border border-green-500/30">
                                <h3 className="text-2xl font-semibold text-green-300 mb-3">
                                    🏛️ Urban Planning Policy
                                </h3>
                                <p className="text-gray-300">
                                    Help city planners and policymakers visualize the impact of development decisions
                                    before implementation.
                                </p>
                            </div>
                            <div className="bg-blue-900/20 p-6 rounded-lg border border-blue-500/30">
                                <h3 className="text-2xl font-semibold text-blue-300 mb-3">
                                    🎓 Academic Research
                                </h3>
                                <p className="text-gray-300">
                                    Support climate research with real-world urban modeling and scenario analysis.
                                </p>
                            </div>
                            <div className="bg-yellow-900/20 p-6 rounded-lg border border-yellow-500/30">
                                <h3 className="text-2xl font-semibold text-yellow-300 mb-3">
                                    🏗️ Real Estate Development
                                </h3>
                                <p className="text-gray-300">
                                    Evaluate climate impact of proposed developments and optimize building placement.
                                </p>
                            </div>
                            <div className="bg-purple-900/20 p-6 rounded-lg border border-purple-500/30">
                                <h3 className="text-2xl font-semibold text-purple-300 mb-3">
                                    🌱 Environmental Advocacy
                                </h3>
                                <p className="text-gray-300">
                                    Demonstrate the importance of green spaces and sustainable urban design.
                                </p>
                            </div>
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Scalability */}
            <section className="px-4 py-20 relative z-10">
                <div className="max-w-4xl mx-auto">
                    <FadeIn delay={0.8}>
                        <h2 className="text-4xl font-bold text-white mb-8 text-center">Scalability</h2>
                        <div className="bg-slate-900/50 p-8 rounded-lg border border-slate-700">
                            <div className="space-y-6">
                                <div>
                                    <h3 className="text-2xl font-semibold text-blue-300 mb-3">
                                        🌐 Global Expansion
                                    </h3>
                                    <p className="text-gray-300">
                                        Built on OpenStreetMap data, the platform can be extended to any city worldwide.
                                        Current focus: Top Indian cities (Bengaluru, Delhi, Mumbai, Chennai) + planned
                                        city comparisons (Dubai, Netherlands).
                                    </p>
                                </div>
                                <div>
                                    <h3 className="text-2xl font-semibold text-green-300 mb-3">
                                        ⚡ Performance Optimized
                                    </h3>
                                    <p className="text-gray-300">
                                        Powered by WebGL and Three.js with GPU acceleration (RTX 3070), handling
                                        thousands of buildings in real-time. Celery task queue for async ML model
                                        execution.
                                    </p>
                                </div>
                                <div>
                                    <h3 className="text-2xl font-semibold text-purple-300 mb-3">
                                        🔌 Modular Architecture
                                    </h3>
                                    <p className="text-gray-300">
                                        Service-oriented design allows easy integration of new ML models, data sources,
                                        and visualization layers. RESTful API for third-party integrations.
                                    </p>
                                </div>
                            </div>
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Team */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.9}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Our Team</h2>
                        <div className="grid md:grid-cols-3 lg:grid-cols-5 gap-6">
                            {teamMembers.map((member, idx) => (
                                <ProfileCard
                                    key={idx}
                                    avatarUrl={member.avatar}
                                    name={member.name}
                                    title={member.branch}
                                    handle={member.usn}
                                    status="active"
                                    contactText="Connect"
                                    enableTilt={true}
                                    showUserInfo={true}
                                />
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* CTA */}
            <section className="px-4 py-20 relative z-10">
                <div className="max-w-4xl mx-auto text-center">
                    <FadeIn delay={1.0}>
                        <h2 className="text-4xl font-bold text-white mb-6">
                            Ready to Explore Urban Climate Impact?
                        </h2>
                        <p className="text-xl text-gray-300 mb-8">
                            Start visualizing how unplanned cities affect our climate and economy
                        </p>
                        <div className="flex gap-4 justify-center flex-wrap">
                            <Button
                                onClick={() => navigate('/dashboard')}
                                size="lg"
                                className="bg-blue-600 hover:bg-blue-700 px-8 py-4 text-lg"
                            >
                                View Dashboard
                            </Button>
                            <Button
                                onClick={() => navigate('/viewer3d')}
                                size="lg"
                                className="bg-green-600 hover:bg-green-700 px-8 py-4 text-lg"
                            >
                                Explore 3D Viewer
                            </Button>
                            <Button
                                onClick={() => navigate('/scenario')}
                                size="lg"
                                className="bg-purple-600 hover:bg-purple-700 px-8 py-4 text-lg"
                            >
                                Build Scenarios
                            </Button>
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Footer */}
            <footer className="px-4 py-8 bg-slate-900 border-t border-slate-700 relative z-10">
                <div className="max-w-6xl mx-auto text-center text-gray-400">
                    <div className="flex items-center justify-center gap-6 mb-4">
                        <GenericLogo className="w-12 h-12 rounded-full overflow-hidden" />
                        <RVCELogo className="w-12 h-12 overflow-hidden" />
                    </div>
                    <p>© 2025 Urban Climate Modeling Platform | Built with React, Three.js, Django</p>
                    <p className="text-sm mt-2">
                        Powered by GraphCast, ClimaX, OpenStreetMap | RTX 3070 GPU Accelerated
                    </p>
                </div>
            </footer>
        </div>
    );
};

export default Homepage;
