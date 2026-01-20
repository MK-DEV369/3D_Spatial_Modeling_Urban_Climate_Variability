import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import FadeIn from '../reactbits/Animations/FadeIn';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Silk from '@/components/ui/Silk';
import ProfileCard from '../ProfileCard';
import VariableProximity from '../VariableProximity';
import Morya from "../Logos/Morya.svg";

const Homepage: React.FC = () => {
    const containerRef = useRef(null);
    const navigate = useNavigate();

    const techStack = [
        { name: 'React + TypeScript', category: 'Frontend Framework' },
        { name: 'Three.js + WebGL', category: '3D Visualization Engine' },
        { name: 'Django + PostGIS', category: 'Spatial Backend' },
        { name: 'GraphCast', category: 'Weather Prediction AI' },
        { name: 'ClimaX Transformer', category: 'Climate Modeling AI' },
        { name: 'ARIMA Model', category: 'Traffic Forecasting' },
        { name: 'PostgreSQL', category: 'Geospatial Database' },
        { name: 'OpenStreetMap API', category: 'Urban Data Source' },
    ];

    const features = [
        {
            title: 'Interactive 3D City Visualization',
            description: 'High-fidelity three-dimensional urban models constructed from OpenStreetMap building footprints with satellite imagery overlay. Utilizes WebGL-accelerated rendering to display thousands of building geometries with real-time manipulation capabilities for scenario planning and comparative analysis.',
        },
        {
            title: 'AI-Powered Climate Prediction',
            description: 'Integrates state-of-the-art machine learning models including GraphCast for short-term weather forecasting and ClimaX transformer architecture for long-term climate projection. Models capture complex atmospheric dynamics and urban heat island effects with high spatial resolution.',
        },
        {
            title: 'Scenario-Based Urban Planning',
            description: 'Enables users to create and compare multiple development scenarios by modifying vegetation coverage, building density, and land use parameters. System quantifies the climate impact of proposed changes through scenario-adjusted feature vectors and predictive modeling.',
        },
        {
            title: 'Traffic and Air Quality Analysis',
            description: 'Employs ARIMA time-series modeling for vehicle count prediction and traffic flow forecasting. Integrates with air quality datasets to analyze correlations between urban mobility patterns, pollutant concentrations, and respiratory health indicators.',
        },
        {
            title: 'Comparative Urban Analysis',
            description: 'Facilitates cross-city comparison between unplanned urban centers and planned developments. Quantifies differences in climate metrics, pollution levels, and infrastructure efficiency to support evidence-based policy recommendations.',
        },
        {
            title: 'Geospatial Data Management',
            description: 'Leverages PostgreSQL with PostGIS extensions for efficient storage and querying of spatial datasets. Supports complex geometric operations, spatial indexing, and real-time data aggregation across user-defined geographic boundaries.',
        },
    ];

    const workflow = [
        { step: 1, title: 'Data Acquisition', desc: 'Automated ingestion of building geometries, road networks, climate records, and mobility datasets from multiple sources' },
        { step: 2, title: 'Spatial Processing', desc: 'Feature extraction, normalization, and 3D model construction using PostGIS geometric operations' },
        { step: 3, title: 'Predictive Modeling', desc: 'Application of ARIMA, transformer-based climate models, and pollution forecasting algorithms' },
        { step: 4, title: 'Scenario Simulation', desc: 'Generation of long-term climate projections under user-defined urban development scenarios' },
    ];

    const applications = [
        {
            title: 'Urban Planning and Policy Development',
            description: 'Provides municipal authorities and urban planners with quantitative tools to evaluate the environmental impact of proposed development projects. Enables evidence-based decision-making by simulating multiple scenarios and comparing long-term climate outcomes before implementation.',
        },
        {
            title: 'Climate Research and Environmental Science',
            description: 'Supports academic researchers in studying micro-climate dynamics, urban heat island effects, and the relationship between urban morphology and atmospheric conditions. Offers a data-driven platform for validating climate models and publishing findings.',
        },
        {
            title: 'Infrastructure Development Assessment',
            description: 'Assists real estate developers and construction firms in evaluating the climate implications of large-scale building projects. Optimizes placement of structures, green spaces, and transportation networks to minimize adverse environmental effects.',
        },
        {
            title: 'Environmental Impact Reporting',
            description: 'Facilitates the preparation of environmental impact assessments for regulatory compliance. Generates detailed reports on temperature variations, air quality projections, and ecosystem disruption associated with proposed developments.',
        },
        {
            title: 'Public Health and Safety Planning',
            description: 'Enables health departments to correlate urban climate conditions with respiratory illness patterns and heat-related mortality. Supports the design of public health interventions and emergency response strategies for extreme weather events.',
        },
        {
            title: 'Education and Public Awareness',
            description: 'Serves as an educational tool for universities and environmental organizations to demonstrate the consequences of unplanned urbanization. Raises public awareness about sustainable development practices through interactive visualizations.',
        },
    ];

    const scalability = [
        {
            title: 'Global Geographic Expansion',
            description: 'The platform architecture is designed to scale horizontally across multiple cities and regions. Built on OpenStreetMap data infrastructure, the system can be deployed for any urban area worldwide with minimal configuration. Current implementation focuses on major Indian metropolitan areas including Bengaluru, Delhi, Mumbai, and Chennai, with comparative analysis against planned cities such as Dubai and Amsterdam.',
        },
        {
            title: 'Computational Performance Optimization',
            description: 'Utilizes GPU-accelerated rendering through Three.js and WebGL to maintain interactive frame rates when displaying complex urban geometries containing tens of thousands of buildings. Backend processing leverages Celery distributed task queues for asynchronous execution of machine learning models, enabling concurrent scenario simulations without blocking user interactions.',
        },
        {
            title: 'Modular Service Architecture',
            description: 'Implements a service-oriented design pattern that decouples data ingestion, spatial processing, machine learning inference, and visualization layers. This architecture facilitates the integration of new predictive models, additional data sources, and third-party climate APIs through standardized RESTful interfaces without requiring modifications to existing components.',
        },
        {
            title: 'Data Storage and Query Efficiency',
            description: 'Employs spatial partitioning and indexing strategies within PostGIS to optimize query performance for large geospatial datasets. Supports efficient range queries, nearest-neighbor searches, and geometric union operations that scale logarithmically with dataset size, ensuring responsive user experience as data volume grows.',
        },
    ];

    const methodology = [
        {
            title: 'Urban Feature Extraction',
            description: 'The system computes a comprehensive set of urban morphology features from three-dimensional city models including building density metrics, height distributions, road network topology, surface coverage ratios, and vegetation indices. These features serve as baseline inputs to predictive models.',
        },
        {
            title: 'Scenario Parametrization',
            description: 'Users define future development scenarios through an interface that allows specification of vegetation changes, construction additions or removals, and demographic projections. These modifications are mapped to quantitative adjustments of the baseline feature vector using the formulation x̃ = x + s.',
        },
        {
            title: 'Multi-Model Prediction Pipeline',
            description: 'Climate forecasting utilizes transformer architectures inspired by ClimaX for capturing long-range temporal dependencies. Traffic prediction employs ARIMA time-series modeling trained on historical mobility data. Air quality forecasting integrates pollutant dispersion models with meteorological predictions.',
        },
    ];

    const teamMembers = [
        {
            name: 'L Moryakantha',
            usn: '1RV24AI406',
            branch: 'Artificial Intelligence and Machine Learning',
            linkedin: 'https://linkedin.com/in/l-morya-kantha',
            github: 'https://github.com/MK-DEV369',
            avatar: Morya
        },
        {
            name: 'Shashank K',
            usn: '1RV23AI',
            branch: 'Artificial Intelligence and Machine Learning',
            linkedin: 'https://linkedin.com/in/member2',
            github: 'https://github.com/member2',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member2'
        },
        {
            name: 'Raghavendra Prasad B M',
            usn: '1RV23CS',
            branch: 'Computer Science and Engineering',
            linkedin: 'https://linkedin.com/in/member3',
            github: 'https://github.com/member3',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member3'
        },
        {
            name: 'Yogendra Ajay Reddy Mareddy',
            usn: '1RV23IS',
            branch: 'Information Science and Engineering',
            linkedin: 'https://linkedin.com/in/member4',
            github: 'https://github.com/member4',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member4'
        },
        {
            name: 'Sathvik Hegde',
            usn: '1RV23IS',
            branch: 'Information Science and Engineering',
            linkedin: 'https://linkedin.com/in/member5',
            github: 'https://github.com/member5',
            avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Member5'
        },
    ];

    return (
        <div className="min-h-screen relative">
            {/* Animated Background */}
            <div className="fixed inset-0 z-0">
                <Silk
                    speed={5}
                    scale={1}
                    color="#87CEEB"
                    noiseIntensity={1.5}
                    rotation={0}
                />
            </div>

            {/* Hero Section */}
            <section className="relative min-h-screen flex items-center justify-center px-4 py-20 z-10">
                <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
                <FadeIn delay={0.2} className="text-center z-10 max-w-5xl">
                    <div className="flex items-center justify-center gap-8 mb-8">
                        {/* <GenericLogo className="w-24 h-24 rounded-full overflow-hidden shadow-2xl" />
                        <RVCELogo className="w-24 h-24 overflow-hidden shadow-2xl" /> */}
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
                    <p className="text-xl md:text-2xl text-blue-200 mb-4 font-light">
                        Quantifying the Climate Impact of Unplanned Urban Development
                    </p>
                    <p className="text-lg text-gray-300 mb-8 max-w-3xl mx-auto text-justify leading-relaxed">
                        A comprehensive artificial intelligence platform that integrates three-dimensional urban modeling, machine learning-based climate prediction, and scenario-based simulation to evaluate how unplanned urbanization affects micro-climate patterns, air quality, and environmental sustainability compared to planned metropolitan areas.
                    </p>
                    <div className="flex gap-4 justify-center flex-wrap items-center">
                        <Button
                            onClick={() => navigate('/dashboard')}
                            size="lg"
                            className="relative overflow-hidden group bg-gradient-to-r from-blue-600 via-cyan-500 to-indigo-600 hover:from-blue-700 hover:via-cyan-600 hover:to-indigo-700 text-white px-10 py-4 text-lg font-semibold rounded-xl shadow-xl shadow-blue-500/20 transition-transform duration-300 hover:scale-105 focus:outline-none focus:ring-4 focus:ring-blue-400/50"
                        >
                            Access Platform Dashboard
                        </Button>
                    </div>
                </FadeIn>
            </section>

            {/* Problem Statement */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.3}>
                        <h2 className="text-4xl font-bold text-white mb-8 text-center">Environmental Challenges of Unplanned Urbanization</h2>
                        <div className="grid md:grid-cols-3 gap-6">
                            <Card className="bg-red-900/30 border-red-500/30">
                                <CardHeader>
                                    <CardTitle className="text-2xl text-red-300">Urban Heat Island Effect</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-gray-300 text-justify leading-relaxed">
                                        Unplanned high-density construction reduces natural ventilation and increases thermal mass, resulting in localized temperature elevations of 2-5°C above surrounding areas. This phenomenon amplifies energy consumption for cooling, exacerbates heat-related health issues, and disrupts local atmospheric circulation patterns.
                                    </p>
                                </CardContent>
                            </Card>
                            <Card className="bg-orange-900/30 border-orange-500/30">
                                <CardHeader>
                                    <CardTitle className="text-2xl text-orange-300">Traffic Congestion and Emissions</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-gray-300 text-justify leading-relaxed">
                                        Inadequate transportation infrastructure planning leads to increased vehicle density, prolonged commute times, and elevated greenhouse gas emissions. Traffic congestion contributes significantly to nitrogen oxide and particulate matter pollution, degrading air quality and public health outcomes.
                                    </p>
                                </CardContent>
                            </Card>
                            <Card className="bg-purple-900/30 border-purple-500/30">
                                <CardHeader>
                                    <CardTitle className="text-2xl text-purple-300">Vegetation Loss and Air Quality</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <p className="text-gray-300 text-justify leading-relaxed">
                                        Rapid urban expansion typically involves removal of vegetation and reduction of permeable surfaces, eliminating natural carbon sinks and air filtration mechanisms. The resulting increase in pollutant concentrations impacts respiratory health and reduces the city's capacity to mitigate climate change effects.
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
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Computational Pipeline</h2>
                        <div className="grid md:grid-cols-4 gap-6">
                            {workflow.map((item, idx) => (
                                <div key={idx} className="relative">
                                    <div className="bg-blue-900/30 p-6 rounded-lg border border-blue-500/30 text-center">
                                        <div className="text-5xl font-bold text-blue-400 mb-3">{item.step}</div>
                                        <h3 className="text-xl font-semibold text-white mb-2">{item.title}</h3>
                                        <p className="text-gray-300 text-sm text-justify leading-relaxed">{item.desc}</p>
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
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Platform Capabilities</h2>
                        <div className="grid md:grid-cols-2 gap-6">
                            {features.map((feature, idx) => (
                                <Card key={idx} className="bg-slate-900/50 border-slate-700 hover:border-slate-500 transition-colors">
                                    <CardHeader>
                                        <CardTitle className="text-xl">{feature.title}</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-gray-300 text-justify leading-relaxed">{feature.description}</p>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Methodology Section */}
            <section className="px-4 py-20 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.6}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Technical Methodology</h2>
                        <div className="space-y-6">
                            {methodology.map((item, idx) => (
                                <Card key={idx} className="bg-slate-900/50 border-slate-700">
                                    <CardHeader>
                                        <CardTitle className="text-xl text-cyan-300">{item.title}</CardTitle>
                                    </CardHeader>
                                    <CardContent>
                                        <p className="text-gray-300 leading-relaxed text-justify">{item.description}</p>
                                    </CardContent>
                                </Card>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Technology Stack */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.7}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Technology Stack</h2>
                        <div className="grid md:grid-cols-4 gap-6">
                            {techStack.map((tech, idx) => (
                                <div
                                    key={idx}
                                    className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 p-6 rounded-lg border border-blue-500/30 text-center"
                                >
                                    <h3 className="text-xl font-semibold text-white mb-1">{tech.name}</h3>
                                    <p className="text-sm text-blue-300">{tech.category}</p>
                                </div>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Use Cases */}
            <section className="px-4 py-20 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.8}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">
                            Platform Applications and Use Cases
                        </h2>
                        <div className="grid md:grid-cols-2 gap-6">
                            {applications.map((app, idx) => (
                                <div key={idx} className="bg-green-900/20 p-6 rounded-lg border border-green-500/30">
                                    <h3 className="text-2xl font-semibold text-green-300 mb-3">
                                        {app.title}
                                    </h3>
                                    <p className="text-gray-300 text-justify leading-relaxed">
                                        {app.description}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Scalability */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={0.9}>
                        <h2 className="text-4xl font-bold text-white mb-8 text-center">Scalability and Performance</h2>
                        <div className="space-y-6">
                            {scalability.map((item, idx) => (
                                <div key={idx} className="bg-slate-900/50 p-8 rounded-lg border border-slate-700">
                                    <h3 className="text-2xl font-semibold text-blue-300 mb-3">
                                        {item.title}
                                    </h3>
                                    <p className="text-gray-300 text-justify leading-relaxed">
                                        {item.description}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Team */}
            <section className="px-4 py-20 relative z-10">
                <div className="max-w-6xl mx-auto">
                    <FadeIn delay={1.0}>
                        <h2 className="text-4xl font-bold text-white mb-12 text-center">Research Team</h2>
                        <div className="grid md:grid-cols-3 gap-8 justify-items-center">
                            {teamMembers.map((member, idx) => (
                                <div key={idx} className="flex flex-col items-center justify-self-center">
                                    <ProfileCard
                                        avatarUrl={member.avatar}
                                        name={member.name}
                                        title={member.branch}
                                        handle={member.usn}
                                        enableTilt={true}
                                        showUserInfo={true}
                                        linkedinUrl={member.linkedin}
                                        githubUrl={member.github}
                                    />
                                </div>
                            ))}
                        </div>
                        <div className="mt-12 text-center">
                            <p className="text-gray-300 text-lg">
                                <span className="font-semibold text-white">Faculty Advisor:</span> Dr. Madhu B R
                            </p>
                            <p className="text-gray-400 mt-2">Department of Electrical and Electronics Engineering</p>
                            <p className="text-gray-400">RV College of Engineering, Bengaluru, India</p>
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* CTA */}
            <section className="px-4 py-20 bg-slate-800/50 relative z-10">
                <div className="max-w-4xl mx-auto text-center">
                    <FadeIn delay={1.1}>
                        <h2 className="text-4xl font-bold text-white mb-6">
                            Evaluate Urban Climate Impact
                        </h2>
                        <p className="text-xl text-gray-300 mb-8 text-justify leading-relaxed mx-auto max-w-3xl">
                            Access the platform dashboard to visualize three-dimensional city models, execute scenario-based climate simulations, and analyze the long-term environmental consequences of urban planning decisions through data-driven predictive modeling.
                        </p>
                        <div className="flex gap-4 justify-center flex-wrap items-center">
                            <Button
                                onClick={() => navigate('/dashboard')}
                                size="lg"
                                className="relative overflow-hidden group bg-gradient-to-r from-blue-600 via-cyan-500 to-indigo-600 hover:from-blue-700 hover:via-cyan-600 hover:to-indigo-700 text-white px-10 py-4 text-lg font-semibold rounded-xl shadow-xl shadow-blue-500/20 transition-transform duration-300 hover:scale-105 focus:outline-none focus:ring-4 focus:ring-blue-400/50"
                            >
                                Launch Dashboard
                            </Button>
                        </div>
                    </FadeIn>
                </div>
            </section>

            {/* Footer */}
            <footer className="px-4 py-8 bg-slate-900 border-t border-slate-700 relative z-10">
                <div className="max-w-6xl mx-auto text-center text-gray-400">
                    <p className="mb-2">
                        © 2025 Urban Climate Modeling Platform | RV College of Engineering
                    </p>
                    <p className="text-sm">
                        Built with React, Three.js, Django, PostgreSQL with PostGIS | Powered by GraphCast, ClimaX, and ARIMA Models
                    </p>
                </div>
            </footer>
        </div>
    );
};

export default Homepage;