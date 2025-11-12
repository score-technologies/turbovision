# Score Vision: A Decentralised Protocol for Real-Time Video Intelligence Markets {#score-vision:-a-decentralised-protocol-for-real-time-video-intelligence-markets}

**Version 1.3 (Draft)** · **Date:** October 26, 2025

## Abstract {#abstract}

This paper presents Score Vision, a novel decentralised protocol built on the Bittensor network that establishes an open, competitive marketplace for real-time video intelligence capabilities. The protocol addresses fundamental challenges in computer vision deployment: the lack of standardised benchmarking for production-ready models, the absence of economic incentives for continuous improvement, and the difficulty of composing specialised vision components into reliable end-to-end systems.

Score Vision introduces a two-tier architecture consisting of atomic **Elements** and composite **Vision Agents**. Elements represent granular, verifiable capabilities (object detection, tracking, calibration) with strict input-output contracts, real-time performance constraints, and multi-dimensional evaluation metrics designed to resist gaming. Vision Agents orchestrate multiple Elements into production-grade pipelines that satisfy end-to-end service level objectives (SLOs) while maintaining temporal consistency and resource efficiency.

The protocol operates through dual execution lanes: a **Public Elements Track** providing transparent benchmarking and open model development, and a **Trusted Agents Track** utilising Trusted Execution Environments (TEEs) to enable privacy-preserving model deployment while maintaining competitive integrity. Both tracks are governed by a cryptographically signed **Manifest** that establishes per-evaluation-window rules, including challenge datasets, evaluation metrics, latency thresholds, and pseudo-ground truth (PGT) generation procedures.

Evaluation quality is ensured through a multi-tier validation architecture that benchmarks new Elements and Agents against state-of-the-art vision-language models (VLMs) and human annotation gold standards. This framework establishes performance baselines, tracks miner convergence toward frontier model capabilities, and validates automated evaluation quality. The system includes provisions for forward reasoning Elements that predict future events based on compositional analysis of upstream outputs, extending the protocol beyond reactive perception toward predictive intelligence.

Our economic model implements a two-phase reward structure. Phase 1 distributes difficulty-weighted emissions based on measurable performance improvements above established baselines, with automatic burning of rewards for underperforming participants. Phase 2 introduces a "Goldilocks Redemption" mechanism that enables USD-denominated revenue from commercial applications to create token redemption pools, establishing a sustainable bridge between decentralised development and commercial deployment.

The protocol incorporates comprehensive anti-gaming mechanisms: per-validator challenge salting using verifiable random functions (VRFs), locally-generated pseudo-ground truth that miners never observe, multi-pillar evaluation metrics, and human-in-the-loop audit systems for quality assurance. Security is further enhanced through TEE attestation, no-egress policies for sensitive computations, and comprehensive telemetry requirements.

Score Vision demonstrates how decentralised networks can create production-ready computer vision capabilities by aligning economic incentives with real-world performance requirements. The protocol's design enables continuous improvement of vision capabilities under real-time constraints while providing a pathway for commercial adoption through Score Cloud, a unified platform offering both public and privacy-preserving execution modes. Our mission is making every camera intelligent through composable, continuously improving vision infrastructure.

---

## Table of Contents {#table-of-contents}

[Score Vision: A Decentralised Protocol for Real-Time Video Intelligence Markets](#score-vision:-a-decentralised-protocol-for-real-time-video-intelligence-markets)

[Abstract](#abstract)

[Table of Contents](#table-of-contents)

[1\. Introduction and Motivation](#1.-introduction-and-motivation)

[1.1 Problem Statement](#1.1-problem-statement)

[1.2 Related Work and Limitations](#1.2-related-work-and-limitations)

[1.3 Our Approach and Contributions](#1.3-our-approach-and-contributions)

[1.4 Broader Vision and Impact](#1.4-broader-vision-and-impact)

[2\. System Architecture and Core Concepts](#2.-system-architecture-and-core-concepts)

[2.1 Elements: Atomic Computational Units](#2.1-elements:-atomic-computational-units)

[2.1.1 Formal Definition and Design Philosophy](#2.1.1-formal-definition-and-design-philosophy)

[2.1.2 Contract Specification and Interface Design](#2.1.2-contract-specification-and-interface-design)

[2.1.3 Real-Time Factor Framework and Hardware-Agnostic Evaluation](#2.1.3-real-time-factor-framework-and-hardware-agnostic-evaluation)

[2.1.4 Multi-Pillar Evaluation Methodology](#2.1.4-multi-pillar-evaluation-methodology)

[2.1.5 Initial Element Specifications](#2.1.5-initial-element-specifications)

[2.1.6 Element Lifecycle and Evolution](#2.1.6-element-lifecycle-and-evolution)

[2.2 Vision Agents: Orchestrated Integration Systems](#2.2-vision-agents:-orchestrated-integration-systems)

[2.2.1 Architectural Overview](#2.2.1-architectural-overview)

[2.2.2 Integration Responsibilities](#2.2.2-integration-responsibilities)

[2.2.3 Performance Contracts and SLO Enforcement](#2.2.3-performance-contracts-and-slo-enforcement)

[2.2.4 Dual-Lane Execution Model](#2.2.4-dual-lane-execution-model)

[2.2.5 Economic Incentive Structure](#2.2.5-economic-incentive-structure)

[2.2.6 Telemetry and Verification Framework](#2.2.6-telemetry-and-verification-framework)

[2.2.7 Versioning and Compatibility Management](#2.2.7-versioning-and-compatibility-management)

[3\. Protocol Architecture and Implementation](#3.-protocol-architecture-and-implementation)

[3.1 System Overview and Design Principles](#3.1-system-overview-and-design-principles)

[3.2 Protocol Execution Flow](#3.2-protocol-execution-flow)

[Phase 1: Manifest Publication and Distribution](#phase-1:-manifest-publication-and-distribution)

[Phase 2: Challenge Request and Distribution](#phase-2:-challenge-request-and-distribution)

[Phase 3: Cryptographic Salt Derivation](#phase-3:-cryptographic-salt-derivation)

[Phase 4: Pseudo-Ground Truth Generation](#phase-4:-pseudo-ground-truth-generation)

[Phase 5: Miner Inference and Submission](#phase-5:-miner-inference-and-submission)

[Phase 6: Validator Scoring and Evaluation](#phase-6:-validator-scoring-and-evaluation)

[Phase 7: Shard Emission and Evidence Recording](#phase-7:-shard-emission-and-evidence-recording)

[Phase 8: Aggregation and Outlier Detection](#phase-8:-aggregation-and-outlier-detection)

[Phase 9: Quality Assurance and Human Audit](#phase-9:-quality-assurance-and-human-audit)

[3.3 Network Participants and Responsibilities](#3.3-network-participants-and-responsibilities)

[3.4 The Manifest: Cryptographic Rule Specification](#3.4-the-manifest:-cryptographic-rule-specification)

[3.5 Challenge API and Distribution](#3.5-challenge-api-and-distribution)

[3.6 Cryptographic Salting Mechanism](#3.6-cryptographic-salting-mechanism)

[3.7 Real-Time Viability Framework](#3.7-real-time-viability-framework)

[3.8 Pseudo-Ground Truth Generation and Validation](#3.8-pseudo-ground-truth-generation-and-validation)

[3.9 Scoring and Aggregation Framework](#3.9-scoring-and-aggregation-framework)

[3.10 Shard Schema (v1.3)](<#3.10-shard-schema-(v1.3)>)

[3.11 Human Audit and Quality Assurance](#3.11-human-audit-and-quality-assurance)

[3.12 Network Architecture and Data Flow](#3.12-network-architecture-and-data-flow)

[3.13 Anti-Gaming Mechanisms and Security](#3.13-anti-gaming-mechanisms-and-security)

[3.14 Evaluation Window Lifecycle](#3.14-evaluation-window-lifecycle)

[3.15 Trusted Execution Environment Integration](#3.15-trusted-execution-environment-integration)

[3.16 Reproducibility and Audit Framework](#3.16-reproducibility-and-audit-framework)

[4\. Evaluation Methodology and Benchmarking Framework](#4.-evaluation-methodology-and-benchmarking-framework)

[4.1 Multi-Tier Validation Architecture](#4.1-multi-tier-validation-architecture)

[4.2 State-of-the-Art VLM Baseline Establishment](#4.2-state-of-the-art-vlm-baseline-establishment)

[4.3 Human Annotation Gold Standard](#4.3-human-annotation-gold-standard)

[4.4 Continuous Miner Performance Tracking](#4.4-continuous-miner-performance-tracking)

[4.5 Rapid Convergence Incentives](#4.5-rapid-convergence-incentives)

[4.6 Forward Reasoning and Predictive Elements](#4.6-forward-reasoning-and-predictive-elements)

[4.6.1 Architectural Foundation](#4.6.1-architectural-foundation)

[4.6.2 Evaluation Methodology](#4.6.2-evaluation-methodology)

[4.6.3 Integration with Perception Elements](#4.6.3-integration-with-perception-elements)

[4.6.4 Phased Deployment Strategy](#4.6.4-phased-deployment-strategy)

[5\. Economic Model and Incentive Mechanisms](#5.-economic-model-and-incentive-mechanisms)

[5.1 Design Principles and Objectives](#5.1-design-principles-and-objectives)

[5.2 Mathematical Framework for Reward Distribution](#5.2-mathematical-framework-for-reward-distribution)

[5.2.1 Emission Surface Partitioning](#5.2.1-emission-surface-partitioning)

[5.2.2 Element-Level Reward Calculation](#5.2.2-element-level-reward-calculation)

[5.2.3 Trusted Track Composite Scoring](#5.2.3-trusted-track-composite-scoring)

[5.3 Phase 1: Baseline Emissions and Burn Mechanisms](#5.3-phase-1:-baseline-emissions-and-burn-mechanisms)

[5.3.1 Latency Gate Enforcement](#5.3.1-latency-gate-enforcement)

[5.3.2 Concentration Guards and Anti-Monopoly Mechanisms](#5.3.2-concentration-guards-and-anti-monopoly-mechanisms)

[5.3.3 Discovery Credit Mechanisms](#5.3.3-discovery-credit-mechanisms)

[5.4 Phase 2: Commercial Revenue Integration ("Goldilocks Redemption")](<#5.4-phase-2:-commercial-revenue-integration-("goldilocks-redemption")>)

[5.4.1 Revenue-Backed Token Redemption](#5.4.1-revenue-backed-token-redemption)

[5.4.2 Redemption Pool Management](#5.4.2-redemption-pool-management)

[5.4.3 Transparency and Reporting](#5.4.3-transparency-and-reporting)

[5.5 Integration with Dynamic TAO Framework](#5.5-integration-with-dynamic-tao-framework)

[6\. Commercialisation Strategy](#6.-commercialisation-strategy)

[6.1 Score Cloud: Unified Platform with Dual Execution Modes](#6.1-score-cloud:-unified-platform-with-dual-execution-modes)

[6.2 Product Layer and Vertical Applications](#6.2-product-layer-and-vertical-applications)

[7\. Security Analysis and Threat Model](#7.-security-analysis-and-threat-model)

[7.1 Threat Landscape Overview](#7.1-threat-landscape-overview)

[7.2 Formal Threat Model](#7.2-formal-threat-model)

[7.2.1 Adversary Capabilities and Objectives](#7.2.1-adversary-capabilities-and-objectives)

[7.2.2 Attack Surface Analysis](#7.2.2-attack-surface-analysis)

[7.3 Specific Attack Vectors and Mitigations](#7.3-specific-attack-vectors-and-mitigations)

[7.3.1 Pre-Computation and Caching Attacks](#7.3.1-pre-computation-and-caching-attacks)

[7.3.2 Latency Gate Circumvention](#7.3.2-latency-gate-circumvention)

[7.3.3 Validator Collusion and Bias Injection](#7.3.3-validator-collusion-and-bias-injection)

[7.3.4 Pseudo-Ground Truth Poisoning](#7.3.4-pseudo-ground-truth-poisoning)

[7.3.5 TEE-Specific Attack Vectors](#7.3.5-tee-specific-attack-vectors)

[7.3.6 Economic Manipulation Attacks](#7.3.6-economic-manipulation-attacks)

[7.4 Residual Risk Assessment](#7.4-residual-risk-assessment)

[7.5 Security Verification and Audit Framework](#7.5-security-verification-and-audit-framework)

[8\. Governance & Upgrades](#8.-governance-&-upgrades)

[8.1 Current Governance Model (v1.3)](<#8.1-current-governance-model-(v1.3)>)

[8.2 Change Management Process](#8.2-change-management-process)

[8.3 Path Toward Decentralisation](#8.3-path-toward-decentralisation)

[9\. Implementation Details](#9.-implementation-details)

[9.1 Element and Agent I/O Schemas](#9.1-element-and-agent-i/o-schemas)

[9.2 Telemetry Requirements](#9.2-telemetry-requirements)

[9.3 Challenge API Specification](#9.3-challenge-api-specification)

[10\. Roadmap](#10.-roadmap)

[10.1 Element Expansion](#10.1-element-expansion)

[10.2 Evaluation Infrastructure Enhancement](#10.2-evaluation-infrastructure-enhancement)

[10.3 Trusted Execution Environment Maturity](#10.3-trusted-execution-environment-maturity)

[10.4 Economic Model Activation](#10.4-economic-model-activation)

[10.5 Operational Transparency](#10.5-operational-transparency)

[11\. Conclusion and Future Directions](#11.-conclusion-and-future-directions)

[11.1 Summary of Contributions](#11.1-summary-of-contributions)

[11.2 Implications for Computer Vision and Potential Methodological Contributions](#11.2-implications-for-computer-vision-and-potential-methodological-contributions)

[11.3 Limitations and Areas for Improvement](#11.3-limitations-and-areas-for-improvement)

[11.4 Future Research Directions](#11.4-future-research-directions)

[11.5 Vision and Impact](#11.5-vision-and-impact)

[11.6 Final Remarks](#11.6-final-remarks)

[Appendix A — JSON Schemas (abbrev)](<#appendix-a-—-json-schemas-(abbrev)>)

[Appendix B — Parameter Tables (defaults)](<#appendix-b-—-parameter-tables-(defaults)>)

[Appendix C — Math](#appendix-c-—-math)

[Appendix D — Manifest Snippet (illustrative)](<#appendix-d-—-manifest-snippet-(illustrative)>)

[Appendix E — Glossary](#appendix-e-—-glossary)

[Appendix F — References](#appendix-f-—-references)

[Computer Vision Models and Datasets](#computer-vision-models-and-datasets)

[Computer Vision Frameworks](#computer-vision-frameworks)

[Industry Platforms](#industry-platforms)

[Trusted Execution Environments](#trusted-execution-environments)

[Decentralised AI Networks](#decentralised-ai-networks)

[Bittensor Subnets](#bittensor-subnets)

[Vision-Language Models](#vision-language-models)

[Annotation and Quality Assurance](#annotation-and-quality-assurance)

[Protocol and Network Infrastructure](#protocol-and-network-infrastructure)

---

## 1\. Introduction and Motivation {#1.-introduction-and-motivation}

### 1.1 Problem Statement {#1.1-problem-statement}

The deployment of computer vision systems in production environments faces a fundamental disconnect between research advances and practical requirements. While individual vision components—object detectors such as YOLO \[[1](#ref1)\] and DETR \[[2](#ref2)\], multi-object trackers like ByteTrack \[[3](#ref3)\] and DeepSORT \[[4](#ref4)\], pose estimation systems including MMPose, and action recognition frameworks such as MMAction—demonstrate impressive performance on standardised benchmarks, their integration into reliable, real-time systems remains a significant engineering challenge that has received insufficient attention from both the research community and industry practitioners.

This gap manifests most acutely in the compositional brittleness of modern vision pipelines. Building production systems requires extensive manual integration work to combine specialised components, each of which may have different input and output formats, processing assumptions, and performance characteristics that were optimised in isolation rather than as part of a larger system. The lack of standardised interfaces means that swapping one component for another—ie, replacing one object detector with a more recent architecture—often requires substantial system redesign, including modifications to preprocessing pipelines, output parsing logic, and downstream components that consume the detector's predictions. This brittleness creates significant friction in the adoption of new research advances and locks production systems into technical debt that accumulates over time.

Domain adaptation presents another critical challenge. Models trained on academic datasets like COCO \[[5](#ref5)\], ImageNet \[[6](#ref6)\], or MOT \[[7](#ref7)\] frequently exhibit severe performance degradation when deployed on real-world data streams that differ in lighting conditions, camera angles, object distributions, or environmental contexts. The domain shift problem is exacerbated by the static nature of training datasets, which cannot capture the dynamic variability of production environments where conditions change seasonally, equipment is upgraded, or new scenarios emerge that were not represented in the original training data. Current approaches to domain adaptation—fine-tuning, transfer learning, or domain-adversarial training—require substantial data collection and retraining efforts that are often impractical in production settings.

Real-time performance constraints create a third dimension of difficulty. Academic benchmarks typically prioritise accuracy over latency, leading to models that achieve state-of-the-art results but are impractical for real-time applications requiring sub-second response times. The trade-offs between accuracy and inference speed are poorly understood and inadequately incentivised in current evaluation frameworks, which report accuracy metrics prominently while relegating timing information to supplementary materials or omitting it entirely. This creates a fundamental misalignment between research incentives and deployment requirements, where practitioners must often choose between using outdated but fast models or accepting unacceptable latency from newer architectures.

Perhaps most fundamentally, deployed vision systems tend to stagnate after initial deployment. There are no systematic mechanisms for continuous model improvement, performance monitoring relative to evolving baselines, or competitive upgrading of system components. Once a system is deployed and integrated into production workflows, the organisational inertia and technical complexity of updates often mean that models remain unchanged for years, even as research advances could offer substantial improvements. This stagnation is not merely a technical problem but reflects the absence of economic structures that would incentivise and facilitate continuous improvement in deployed systems.

### 1.2 Related Work and Limitations {#1.2-related-work-and-limitations}

Various existing approaches attempt to address subsets of these challenges, but each suffers from limitations that prevent comprehensive solutions to the production deployment problem. Model serving platforms such as TensorFlow Serving, TorchServe, and MLflow have emerged to standardise the infrastructure layer of model deployment, providing APIs for model loading, versioning, and inference serving. While these platforms reduce the operational burden of deploying individual models, they do not address the fundamental problems of component composition, performance optimisation under resource constraints, or continuous improvement through competitive development. They assume that model selection and optimisation happen elsewhere and simply provide the infrastructure to serve whatever models practitioners have already developed and validated.

Computer vision frameworks represent another category of solutions. Comprehensive frameworks such as OpenMMLab \[[9](#ref9)\], Detectron2 \[[8](#ref8)\], and PaddlePaddle \[[10](#ref10)\] offer standardised implementations of vision algorithms with consistent APIs and modular architectures that facilitate experimentation and comparison. These frameworks have significantly accelerated research by providing high-quality reference implementations and reducing the barrier to entry for developing new models. However, they lack mechanisms for real-time performance validation under production constraints or economic incentives that would drive continuous improvement. They are primarily research tools designed to facilitate publication and reproducibility rather than production deployment systems designed to ensure reliability and performance at scale.

Industry platforms for intelligent video analytics have emerged to address production deployment more directly. NVIDIA Metropolis \[[11](#ref11)\] provides vision AI microservices and workflows (video ingest/storage, inference via NIM \[[13](#ref13)\], multi-camera tracking with DeepStream \[[12](#ref12)\]) and fine-tuning via TAO's \[[14](#ref14)\] vision foundation models. These platforms successfully address many production deployment challenges—performance optimisation through hardware-software co-design, standardised deployment workflows, and ecosystem integration—making them valuable for organisations with the resources to adopt proprietary technology stacks. However, they operate as closed or semi-closed ecosystems where model development and improvement happen primarily within the vendor's organisation or through curated partnerships. There is no open competitive mechanism driving continuous improvement, no transparent benchmarking allowing external validation of capability claims, and limited ability for independent developers to contribute innovations or compete on equal footing. The platforms provide excellent solutions for deploying existing capabilities but lack the dynamic, competitive pressure that drives rapid capability advancement.

The space of decentralised AI networks represents a more recent development. Fetch.ai \[[19](#ref19)\], SingularityNET \[[20](#ref20)\], and Ocean Protocol \[[21](#ref21)\] focus on agent/service/data marketplaces, not real-time, latency-gated CV evaluation. While these projects recognise the potential for economic incentives to drive AI development and the value of open competition, they have not addressed the unique challenges of real-time computer vision—such as the need for strict latency guarantees, the importance of composability across different perception tasks, or the difficulty of evaluation under hardware constraints.

Benchmark competitions and challenges—including COCO for object detection, ImageNet for classification, MOT for multi-object tracking, and numerous domain-specific challenges—have been instrumental in driving computer vision research forward by providing standardised evaluation protocols and fostering competition among research groups. These benchmarks have successfully motivated the development of increasingly accurate models and have created shared reference points for comparing different approaches. However, they evaluate models in isolation rather than as components of integrated systems, typically using offline datasets rather than streaming data, and they lack the continuous, real-time evaluation necessary for production deployment. Perhaps most critically, once a challenge concludes, there is no ongoing mechanism for continued improvement or for translating winning solutions into production systems that can be reliably deployed and maintained.

### 1.3 Our Approach and Contributions {#1.3-our-approach-and-contributions}

Score Vision addresses these limitations through a novel decentralised protocol that creates economic incentives for developing production-ready computer vision components while maintaining the open, competitive dynamics that drive innovation in research settings. Rather than attempting to solve all deployment challenges through centralised infrastructure or hoping that research advances will eventually trickle down to production systems, we design mechanisms that align economic incentives with practical deployment requirements and enable continuous, competitive improvement of vision capabilities under real-world constraints.

Our first major contribution is the introduction of atomic Elements as the fundamental unit of capability development and evaluation. Elements represent granular, verifiable capabilities—such as object detection, tracking, or calibration—with strict input-output contracts that specify exactly what data they consume and produce. This design enables modular development where specialised teams or individuals can focus on improving specific capabilities without requiring deep knowledge of the entire system. Critically, Elements can be continuously benchmarked against both automated baselines and human annotations, and they can be composed into larger systems while maintaining system-level performance guarantees through explicit interface contracts and telemetry requirements.

The second contribution addresses the pervasive disconnect between accuracy-focused research and latency-constrained deployment. Unlike traditional benchmarks that treat latency as a secondary concern or report it separately from accuracy metrics, our protocol makes real-time performance a first-class constraint through Real-Time Factor (RTF) gates that enforce practical viability. The RTF framework provides hardware-agnostic evaluation by measuring performance relative to declared service rates rather than absolute timing on specific hardware configurations. This approach ensures that competitive pressure drives genuine algorithmic efficiency improvements rather than simply rewarding participants with access to more powerful hardware, while still maintaining the constraint that solutions must be practically deployable in real-world systems.

Third, we introduce a dual-lane execution model that reconciles the tension between open development and commercial privacy requirements. The Public Elements Track operates with full transparency, enabling open benchmarking, model sharing, and collaborative improvement that has proven so valuable in research settings. The Trusted Agents Track utilises Trusted Execution Environments (TEEs) to enable privacy-preserving deployment where models and weights remain confidential within attested enclaves, addressing the legitimate commercial concerns that prevent many organisations from participating in open development. Both tracks operate under identical evaluation criteria and contribute to the same competitive ecosystem, enabling a smooth transition from research to production deployment without compromising the competitive dynamics that drive innovation.

Fourth, we design a token-based economic model that pays only for measurable improvements above established baselines, creating sustainable incentives for continuous model enhancement while preventing reward extraction through gaming or manipulation. The economic mechanism incorporates difficulty weighting that tilts rewards toward harder problems, baseline gates that burn emissions for underperforming participants, and discovery credits that reward miners who identify errors in automated evaluation systems. This structure ensures that economic incentives align with genuine capability improvements rather than clever exploitation of evaluation quirks.

Finally, we incorporate comprehensive security measures to prevent manipulation while preserving openness. The protocol uses per-validator challenge salting with verifiable random functions to prevent pre-computation attacks, locally-generated pseudo-ground truth that miners never observe, multi-pillar evaluation metrics designed to resist single-dimension optimisation, and human-in-the-loop audit systems that validate automated evaluations and enable discovery of systematic biases. These mechanisms work together to create an environment where the most reliable path to rewards is genuine capability improvement rather than gaming or collusion.

### 1.4 Broader Vision and Impact {#1.4-broader-vision-and-impact}

The ultimate goal of Score Vision is to make every camera intelligent by creating decentralised infrastructure for production-ready computer vision. Current AI development follows a pattern where research produces impressive capabilities on benchmark tasks, but the gap between research demonstrations and production-ready systems remains stubbornly large. This gap is not primarily technical—the algorithms and architectures needed for many applications already exist—but rather stems from misaligned incentives, lack of continuous improvement mechanisms, and the absence of economic structures that reward production readiness over benchmark performance.

By creating a sustainable economic model for production-ready AI components that explicitly rewards real-time performance, composability, and continuous improvement, we aim to demonstrate an alternative development paradigm. In this paradigm, economic incentives drive capabilities toward practical utility rather than academic novelty, open competition produces continuous improvement rather than one-time benchmark victories, and standardised interfaces enable reliable composition rather than brittle integration. The principles underlying Score Vision—atomic capability units, real-time performance gates, automated evaluation with human validation, and economic incentives for improvement—are not specific to computer vision but could be applied to other AI domains where the research-deployment gap remains problematic.

This approach directly addresses the fundamental problems identified in Section 1.1. Compositional brittleness is solved through standardised Element interfaces with strict I/O contracts, enabling reliable system integration where components can be swapped without requiring downstream modifications. The domain shift problem is mitigated through continuous competitive evaluation on diverse, real-world data streams rather than static benchmark datasets, creating economic incentives for models that generalise across varying conditions rather than overfit to specific distributions. The absence of continuous improvement mechanisms is resolved through the protocol's economic structure, where ongoing emissions reward sustained performance improvements and discovery credits incentivize miners to identify and correct evaluation system limitations, creating a feedback loop that drives perpetual capability advancement rather than one-time benchmark victories.

The vision of real intelligence operating in dynamic, real-world environments requires robust perception capabilities that can adapt to changing conditions, maintain performance under resource constraints, and improve continuously as new challenges emerge. Static datasets and offline evaluation metrics, while valuable for research, are fundamentally insufficient for this challenge because they cannot capture the temporal dynamics, environmental variability, and operational constraints of actual deployment. Score Vision transforms computer vision development from a process of training models on fixed datasets toward an ongoing competition that operates under the time pressure and resource constraints characteristic of production systems. This transformation represents not merely an engineering improvement but a fundamental reimagining of how AI capabilities can be developed, evaluated, and deployed at scale.

---

## 2\. System Architecture and Core Concepts {#2.-system-architecture-and-core-concepts}

### 2.1 Elements: Atomic Computational Units {#2.1-elements:-atomic-computational-units}

#### 2.1.1 Formal Definition and Design Philosophy {#2.1.1-formal-definition-and-design-philosophy}

Elements represent the fundamental abstraction at the heart of the Score Vision protocol: atomic, verifiable computational capabilities that can be independently developed, continuously evaluated, and reliably composed into larger systems. Unlike monolithic vision pipelines where capabilities are tightly coupled and difficult to isolate, Elements embody the principle that complex perception tasks can be decomposed into well-defined, specialised functions that maintain strict interface contracts while competing for continuous improvement.

Formally, we define an Element E as a six-tuple E \= (I, O, C, L, M, P), where I specifies the input schema that the Element must consume, O defines the structured output format it produces, C establishes the performance contract including declared service rates and resource bounds, L implements the latency gate function that enforces real-time viability, M defines the multi-dimensional metric evaluation function used to assess quality, and P specifies the pre-processing pipeline that ensures consistent input normalisation. This tuple captures everything necessary to implement, evaluate, and deploy an Element: what it processes, what it produces, how fast it must operate, how its quality is measured, and how its inputs are prepared.

The design philosophy underlying Elements prioritises several key properties that distinguish them from traditional computer vision components. First, atomicity ensures that each Element performs a single, well-scoped capability—detecting objects, tracking entities, calibrating geometry—rather than attempting to solve multiple tasks simultaneously. This specialisation enables focused optimisation and makes capability boundaries explicit. Second, verifiability requires that Element outputs follow standardised JSON schemas that can be programmatically validated and scored against reference annotations, eliminating ambiguity about what constitutes correct behavior. Third, composability demands that Elements operate as pure functions transforming standardised inputs to standardised outputs, with no hidden state or side effects that would complicate integration into larger systems.

Each Element operates within strict performance and accuracy constraints that ensure practical deployability. The performance contract specifies not just what the Element must compute but how quickly it must do so relative to real-time requirements, creating economic pressure toward algorithmic efficiency rather than simply throwing more compute at problems. The multi-dimensional evaluation captures different aspects of quality—spatial accuracy, temporal consistency, completeness, semantic correctness—preventing optimisation toward single metrics that might not reflect genuine capability improvements. The atomic nature of Elements ensures they can be independently developed by specialised teams or individuals without requiring deep knowledge of the entire system architecture, while standardised interfaces guarantee that improvements in one Element automatically benefit any system that composes it.

#### 2.1.2 Contract Specification and Interface Design {#2.1.2-contract-specification-and-interface-design}

The Element contract establishes a formal interface that enables reliable composition and evaluation while providing clear expectations for both implementers and consumers. This contract serves multiple purposes simultaneously: it defines what data flows between components, establishes performance requirements that ensure real-time viability, specifies how quality will be measured, and creates a stable foundation for long-term system evolution. Unlike informal interface definitions that rely on documentation and convention, Element contracts are machine-readable specifications that can be programmatically validated and enforced.

Input and output schemas form the syntactic foundation of Element contracts. All Elements consume standardised `FrameBatch` objects containing temporal sequences of video frames represented as URLs or byte arrays, along with associated metadata including timestamps, resolution, frame rate, and any domain-specific context required for processing. This standardisation ensures that any Element can be swapped for another Element serving the same function without modifying upstream data sources. Output formats vary by Element type but follow consistent JSON schema patterns that enable downstream consumption and evaluation. Detection Elements produce `Detections[]` arrays containing bounding boxes with normalised coordinates, confidence scores, tracking identifiers that maintain temporal consistency, and optional semantic attributes such as team assignments or role classifications. Calibration Elements output `Keypoints[]` arrays with spatial coordinates for detected geometric features along with homography matrices that enable transformation between image space and real-world coordinate systems. Coordinate Elements generate `XY[]` arrays mapping tracked entities to field-space positions with explicit units and coordinate frame specifications.

Performance contracts extend beyond mere functional correctness to specify the temporal and computational requirements that make Elements deployable in production systems. Each Element declares a service rate re representing the frames-per-second it must effectively support when deployed, creating a binding commitment that enables system-level performance planning and resource allocation. This declaration is not merely aspirational documentation but forms the basis for the Real-Time Factor evaluation that gates economic rewards. Elements must also declare their resource requirements—typical GPU memory consumption, batch size preferences, and any special hardware dependencies—enabling deployment systems to provision appropriate infrastructure and schedule workloads efficiently.

Latency constraints operationalise the performance contract through explicit, measurable requirements. Elements must satisfy strict real-time performance requirements expressed through percentile latency bounds, specifically requiring that the 95th percentile latency for processing a canonical 5-FPS batch not exceed Element-specific thresholds typically ranging from 200 to 250 milliseconds. The focus on p95 rather than mean or median latency recognises that production systems must handle worst-case performance, not just typical cases, and that occasional slow frames can disrupt downstream processing or user experience. These thresholds are calibrated against the declared service rates to ensure that Elements can genuinely meet their claimed real-time performance rather than merely achieving good average-case behavior.

Evaluation metrics complete the contract by specifying how Element quality will be assessed during competitive evaluation. Elements are assessed through multi-pillar evaluation functions that capture different aspects of performance quality—spatial accuracy of detections, temporal consistency of tracking, completeness of coverage, semantic correctness of classifications—with explicit weights assigned to each pillar. This multi-dimensional approach prevents gaming through single-metric optimisation while ensuring comprehensive capability assessment. The evaluation specification includes not just the metrics themselves but also the reference generation procedure (how pseudo-ground truth is created), the comparison methodology (how miner outputs are matched to references), and the aggregation logic (how per-frame scores combine into overall quality measures).

#### 2.1.3 Real-Time Factor Framework and Hardware-Agnostic Evaluation {#2.1.3-real-time-factor-framework-and-hardware-agnostic-evaluation}

The Real-Time Factor (RTF) framework addresses a fundamental challenge in evaluating vision systems: how to assess real-time performance in a way that is fair across different hardware configurations while still ensuring practical deployability. Traditional approaches to performance evaluation either measure absolute latency on specific hardware, which unfairly advantages participants with access to expensive accelerators, or ignore latency entirely, which fails to ensure that winning solutions can actually meet production requirements. The RTF framework resolves this tension by measuring performance relative to declared service rates rather than absolute time, creating hardware-agnostic evaluation that still enforces real-time viability.

For an Element E with measured p95 batch processing time t_p95 (in milliseconds) and declared service rate r_e (in frames per second), the Real-Time Factor is calculated as RTF_e \= (t_p95 / 1000\) × (r_e / 5). This formula captures the relationship between actual processing speed and required throughput in a normalised form. The denominator of 5 reflects the canonical evaluation sampling rate at which all Elements are benchmarked for comparability, while the service rate r_e in the numerator establishes the target throughput the Element must support in production. An Element passes the real-time gate if RTF_e ≤ 1.0, indicating that it can process frames at least as fast as its declared service rate requires.

This formulation creates several desirable properties. First, it enables fair comparison across heterogeneous hardware configurations—a solution running on a high-end GPU and achieving 100ms p95 latency competes on equal footing with a solution running on edge hardware achieving 400ms latency if both meet their respective RTF requirements. Second, it prevents gaming through hardware scaling by tying rewards to algorithmic efficiency rather than raw compute power. Third, it accommodates different use cases that have different real-time requirements—calibration Elements that need only run at 5 FPS face more lenient latency bounds than detection Elements that must support 25 FPS, reflecting actual deployment needs. Fourth, it maintains practical deployment viability by ensuring that economic rewards flow only to solutions that can genuinely operate in real-time on realistic hardware, preventing the emergence of impractical solutions that achieve high accuracy through excessive computation.

#### 2.1.4 Multi-Pillar Evaluation Methodology {#2.1.4-multi-pillar-evaluation-methodology}

Single-metric optimisation has repeatedly proven inadequate for assessing computer vision systems in production contexts. Models that achieve excellent performance on one dimension—say, spatial accuracy of bounding boxes—may fail catastrophically on other critical dimensions such as temporal consistency, completeness, or semantic correctness. The multi-pillar evaluation methodology addresses this limitation by decomposing Element quality into multiple independent dimensions, each capturing a distinct aspect of performance that matters for practical deployment. Elements receive rewards based on weighted combinations of these pillars, creating incentives to achieve balanced improvement across all aspects of capability rather than gaming specific metrics.

For object detection and tracking Elements, evaluation pillars capture both spatial and temporal aspects of performance. Spatial accuracy is measured through Intersection over Union (IoU) metrics computed at multiple threshold levels, revealing how precisely Elements localise objects in image space. Count accuracy measures detection completeness by penalising both false positives (phantom detections) and false negatives (missed objects), ensuring that Elements capture the full scene content rather than cherry-picking easy cases. Team palette symmetry, relevant for sports applications, assesses whether Elements correctly classify object attributes such as team assignments based on visual features like uniform colors. Identity switch penalties enforce temporal consistency by detecting when tracking IDs are incorrectly reassigned across frames, a common failure mode that disrupts downstream analysis. Role classification accuracy, where applicable, measures semantic understanding beyond mere detection, requiring Elements to distinguish between functionally different object types such as goalkeepers versus field players.

Keypoint detection and calibration Elements face different evaluation criteria reflecting their geometric nature. Reprojection error forms the primary spatial accuracy metric, with exponential decay scoring (score \= exp(-error/σ)) that heavily penalizes large errors while tolerating small deviations that have minimal practical impact. Temporal stability across frame sequences captures the consistency of keypoint detections over time, penalising jitter or sudden discontinuities that would disrupt camera calibration or motion analysis. Geometric consistency with known constraints leverages domain knowledge—such as the fixed dimensions of sports fields or the parallelism of certain lines—to assess whether detected keypoints satisfy expected spatial relationships. Coverage metrics measure keypoint completeness, ensuring that Elements detect sufficient geometric features to enable reliable calibration even when some keypoints are occluded or ambiguous.

Coordinate estimation Elements that map image-space detections to real-world positions face evaluation criteria that emphasize both accuracy and consistency across spatial and temporal dimensions. Field-space coordinate accuracy directly measures the precision of position estimates in physical units, the ultimate metric for applications requiring absolute positioning. Temporal continuity of position estimates assesses smoothness and physical plausibility of motion, penalising discontinuous jumps that violate kinematic constraints. Spatial coverage and boundary handling evaluate whether Elements maintain accuracy across the full field of play, including challenging edge cases near image borders or in distant regions where resolution is limited. Multi-camera consistency, where applicable, measures whether Elements produce coherent position estimates when the same scene is observed from different viewpoints, a critical requirement for systems that fuse information from multiple camera angles.

#### 2.1.5 Initial Element Specifications {#2.1.5-initial-element-specifications}

The protocol launches with three foundational Elements designed for sports video analysis, chosen to span different capability types and difficulty levels while addressing real production needs. These initial Elements establish the evaluation framework and demonstrate how the Element abstraction applies to diverse vision tasks. The specifications reflect careful calibration of performance requirements, difficulty weights, and evaluation metrics based on both technical feasibility and market demand for these capabilities.

PlayerDetect_v1 serves as the baseline Element for object detection and tracking, focusing on identifying and following player entities across video frames with semantic role classification. This Element must operate at 25 FPS service rate with p95 latency not exceeding 200ms per 5-FPS evaluation batch, establishing a demanding real-time requirement that reflects the need for high-temporal-resolution tracking in sports analytics. The difficulty weight β is set to 1.0, providing a reference point against which other Elements are calibrated. PlayerDetect outputs include bounding boxes localising each player in image coordinates, persistent tracking IDs that maintain identity across frames, team assignments based on visual features like uniform colors, and role classifications distinguishing functional player types. The evaluation combines spatial accuracy (IoU), count accuracy, team classification correctness, tracking continuity, and role classification accuracy with weights reflecting their relative importance for downstream applications.

BallDetect_v1 addresses the notably more challenging task of detecting and tracking small, fast-moving objects that may exhibit significant motion blur and frequent occlusions. Operating at the same 25 FPS service rate and 200ms latency constraint as PlayerDetect, this Element faces the additional difficulty of reasoning about objects that may occupy only a few dozen pixels and move several image-widths between frames. The elevated difficulty weight of β \= 1.4 reflects this increased complexity and creates stronger economic incentives for developing solutions to this harder problem. BallDetect outputs focus on precise spatial localisation, temporal tracking continuity even through brief occlusions, and motion vector estimates that enable prediction of ball trajectory for downstream reasoning tasks. Evaluation emphasizes both instantaneous detection accuracy and temporal consistency, with particularly stringent requirements for maintaining track identity through challenging conditions.

PitchCalib_v1 performs geometric calibration by detecting field keypoints and estimating homography matrices that map between image space and real-world field coordinates. Unlike the detection Elements, calibration can operate at lower temporal resolution since field geometry remains relatively stable across frames. The specified 5 FPS service rate reflects this reduced temporal requirement, and the p95 latency constraint is relaxed to 250ms to accommodate the more complex geometric reasoning involved. The difficulty weight of β \= 0.9, slightly below baseline, acknowledges that calibration generally requires less frequent updates than frame-by-frame detection, though the geometric reasoning itself remains challenging. Outputs include detected keypoint locations for field markers such as corners and line intersections, full 3×3 homography matrices enabling bi-directional transformation between coordinate systems, and confidence metrics that downstream systems can use to assess calibration quality. Evaluation focuses on reprojection accuracy, geometric consistency with known field constraints, and temporal stability of calibration across frames.

#### 2.1.6 Element Lifecycle and Evolution {#2.1.6-element-lifecycle-and-evolution}

The introduction of new Elements follows a structured development and deployment process designed to ensure technical viability, economic sustainability, and protocol compatibility before full production activation. This lifecycle balances the need for rapid capability expansion with the requirement for careful validation that new Elements can be reliably evaluated and will attract meaningful miner participation.

The process begins with feasibility assessment, where protocol governance evaluates both technical and economic viability of proposed Elements. Technical feasibility considers whether the capability can be reliably automated, whether reference annotations can be generated or acquired at reasonable cost, and whether evaluation metrics can capture quality in ways that resist gaming. Economic viability assesses market demand for the capability, the difficulty level relative to existing Elements, and whether the economic incentives (difficulty weights, baseline thresholds) can be calibrated to attract miner participation while rewarding genuine improvements. This assessment phase filters proposals to ensure that development resources focus on Elements that will meaningfully expand protocol capabilities.

Contract definition formalizes the Element specification through precise documentation of input-output schemas, performance requirements, and evaluation metrics. This phase produces machine-readable specifications that can be programmatically validated, ensuring that all participants—miners implementing the Element, validators evaluating it, and downstream consumers utilising it—share identical understanding of requirements and expectations. The contract includes not just functional specifications but also the economic parameters (difficulty weights, baseline thresholds) that will govern reward distribution and the operational parameters (latency gates, service rates) that ensure practical deployability.

PGT recipe development creates the reference annotation generation procedures that validators will use to evaluate miner outputs. This critical phase determines evaluation quality and fairness. The recipe must be deterministic (producing identical outputs given identical inputs and random seeds), reproducible (enabling third-party audit), and high-quality (reflecting genuine ground truth as validated through human annotation comparison). Recipe development typically involves extensive experimentation with vision-language model ensembles, validation against human annotations, and calibration of confidence thresholds and consistency requirements. The finalised recipe is cryptographically hashed and included in the Element contract, binding validators to specific evaluation procedures.

Manifest integration incorporates the new Element into the evaluation framework by adding it to evaluation window specifications with initial parameter settings. This phase includes technical integration testing to ensure that the challenge API, evaluation pipelines, and aggregation systems correctly handle the new Element type. Initial parameter settings (difficulty weights, baselines, metric weights) are conservatively calibrated based on preliminary testing, with the understanding that they may be adjusted as real miner participation provides better data about capability difficulty and performance distributions.

Shadow evaluation runs the new Element in parallel with established Elements but without economic impact, enabling protocol operators to validate evaluation procedures, miners to develop and test implementations, and governance to calibrate economic parameters based on observed performance distributions. During shadow evaluation, all evaluation mechanics operate normally—challenges are distributed, miner submissions are scored, shards are published—but no rewards are distributed and Element performance does not affect miner economics. This phase typically lasts multiple evaluation windows, providing sufficient time for the miner population to develop competitive solutions and for protocol operators to identify and resolve any evaluation issues.

Production activation transitions the Element to full economic participation, enabling miners to earn rewards based on their performance. Activation requires that shadow evaluation has demonstrated stable operation, that multiple competitive miner implementations exist (ensuring that competition will drive improvement), and that economic parameters have been calibrated to produce reasonable reward distributions. Once activated, the Element participates fully in emissions distribution according to its difficulty weight and miner performance levels. Each new Element undergoes comprehensive baseline benchmarking as detailed in Section 4, establishing VLM performance ceilings and human annotation gold standards before production deployment, ensuring that automated evaluation remains grounded in measurable quality standards.

### 2.2 Vision Agents: Orchestrated Integration Systems {#2.2-vision-agents:-orchestrated-integration-systems}

#### 2.2.1 Architectural Overview {#2.2.1-architectural-overview}

Vision Agents represent the second tier of the Score Vision architecture, addressing the gap between atomic capabilities and production-ready systems. While Elements prove that specific capabilities can be continuously improved through competition, real-world applications require integrated pipelines that coordinate multiple Elements, manage shared resources, maintain temporal consistency, and satisfy end-to-end performance guarantees.

Formally, a Vision Agent A is defined as the tuple (E_set, S, F, T, R), where E_set specifies the Elements being composed, S defines the scheduling strategy, F implements output fusion, T provides telemetry proving performance claims, and R enforces resource management and SLO compliance. This captures the essential Agent responsibilities: which Elements to use, when and how to execute them, how to fuse their outputs, how to prove performance requirements are met, and how to ensure resource consumption remains bounded.

#### 2.2.2 Integration Responsibilities {#2.2.2-integration-responsibilities}

Agents address system-level challenges that transcend individual Element boundaries. Temporal orchestration maintains consistent entity identities across frame sequences and camera transitions, managing tracking ID assignments, handling occlusions, and ensuring smooth handoffs when the same entity is tracked by different Elements or across temporal gaps. Resource optimisation maximizes encoder reuse across Elements sharing similar feature extraction needs, optimises batch sizes to fully utilise GPU parallelism, and minimizes memory footprint while maintaining real-time performance.

Output fusion combines predictions when multiple Elements contribute to the same semantic understanding—for instance, fusing Player detections with Pitch calibration to generate Field coordinates requires geometric reasoning about how image-space detections map to real-world positions. Quality assurance enforces system-level invariants such as geometric consistency (ensuring fused outputs respect physical constraints), temporal smoothness (preventing discontinuous jumps violating physics), and physical plausibility (rejecting outputs implying impossible velocities).

#### 2.2.3 Performance Contracts and SLO Enforcement {#2.2.3-performance-contracts-and-slo-enforcement}

Agents operate under strict Service Level Objectives ensuring production readiness. The primary constraint requires RTF ≤ 1.0 relative to aggregate Element service rates, ensuring orchestration overhead does not compromise real-time viability. Jitter bounds limit processing time variance to below 40ms for predictable system behavior. Memory management constrains peak GPU consumption (default 6000MB, configurable per window) with continuous monitoring to prevent resource exhaustion. Deterministic behavior requirements mandate consistent outputs for identical inputs, with pre-processing versions pinned in Manifests for reproducibility.

#### 2.2.4 Dual-Lane Execution Model {#2.2.4-dual-lane-execution-model}

The protocol reconciles open development with commercial privacy through parallel execution tracks. The Public Agents Track operates with full transparency—code, model weights, and intermediate outputs are accessible for analysis and improvement, enabling collaborative development. The Trusted Agents Track utilizes TEEs for privacy-preserving deployment where models and weights remain confidential within attested enclaves, with only structured outputs and telemetry escaping. Both tracks evaluate identical challenges under identical performance requirements, ensuring privacy does not compromise competitive integrity.

#### 2.2.5 Economic Incentive Structure {#2.2.5-economic-incentive-structure}

Agent incentives complement rather than replace Element-level rewards. Individual Elements continue earning based on atomic performance, preventing Agent integration from distorting Element-level incentives. Agents passing end-to-end integration tests—demonstrating RTF ≤ 1.0, jitter ≤ 40ms, ID continuity ≥ 0.95, and complete telemetry validation—earn a 5% bonus on constituent Elements' earnings. This modest bonus incentivises integration excellence without overwhelming pressure to bundle Elements prematurely.

The Trusted Agents Track receives dedicated emission allocation (γ_trusted share) distributed via composite scores weighting Player performance (35%), Ball performance (35%), Calibration performance (20%), and integration quality (10%). This recognises that privacy-preserving deployment serves distinct market needs meriting separate reward allocation.

#### 2.2.6 Telemetry and Verification Framework {#2.2.6-telemetry-and-verification-framework}

All Agent performance claims require comprehensive telemetry validation. Performance metrics include detailed timing distributions (p50, p95, maximum), jitter measurements, and throughput statistics enabling validators to confirm RTF and jitter requirements. Resource utilisation telemetry tracks GPU and CPU memory peaks, encoder reuse ratios demonstrating efficient resource sharing, and batching efficiency metrics. Quality indicators measure ID stability rates, identity switch penalties, entity drop rates, and continuity across temporal sequences. For Trusted Agents, security compliance telemetry includes attestation bundles proving TEE integrity, policy compliance verification, and no-egress confirmation.

#### 2.2.7 Versioning and Compatibility Management {#2.2.7-versioning-and-compatibility-management}

Agents follow semantic versioning (MAJOR.minor.patch) with strict compatibility rules to enable stable composition and controlled evolution. Interface stability rules permit MAJOR version changes to modify I/O schemas when necessary, while minor and patch versions must maintain backward compatibility. Each evaluation window's Manifest specifies eligible Agent versions through allow-lists, enabling controlled rollout while preventing untested versions from affecting rewards. Agents must explicitly declare Element dependencies and version requirements, enabling automatic compatibility verification that prevents deployment of incompatible Agent-Element combinations.

---

## 3\. Protocol Architecture and Implementation {#3.-protocol-architecture-and-implementation}

### 3.1 System Overview and Design Principles {#3.1-system-overview-and-design-principles}

Score Vision implements a decentralised validation network on Bittensor built around four core architectural principles.

1. First, separation of concerns cleanly partitions evaluation logic (Elements/Agents), economic incentives (emission mechanisms), and governance (Manifest management), enabling independent evolution of each component.
2. Second, cryptographic verifiability ensures all protocol operations produce signed evidence that can be independently verified, enabling transparency and dispute resolution.
3. Third, privacy-preserving competition reconciles open development with commercial requirements through dual execution lanes operating under identical rules.
4. Fourth, real-time constraints are enforced as first-class requirements rather than secondary metrics, ensuring practical deployment viability.

The system operates as a two-lane validation network where transparent Public Elements and privacy-preserving Trusted Agents compete in parallel evaluation cycles. Cryptographically signed Manifests establish per-window rules, validators build pseudo-ground truth locally, miners never observe ground truth, and all evidence is publicly auditable through signed shards.

### 3.2 Protocol Execution Flow {#3.2-protocol-execution-flow}

The protocol operates through structured evaluation cycles, each governed by a time-bounded window with deterministic rules. Nine phases constitute each cycle:

#### Phase 1: Manifest Publication and Distribution {#phase-1:-manifest-publication-and-distribution}

Each window begins with publication of a cryptographically signed Manifest establishing all rules, constraints, and parameters. The Manifest is content-addressed via SHA-256 hashing and distributed through decentralised storage, specifying challenge datasets and weights, Element-specific metrics and thresholds, pre-processing parameters, latency gates and service rates, PGT recipe specifications, economic parameters (baselines, difficulty weights, burn rates), and eligibility criteria for participants.

#### Phase 2: Challenge Request and Distribution {#phase-2:-challenge-request-and-distribution}

Validators request challenges through the Challenge API using cryptographically authenticated requests that include the Manifest hash for version consistency. The API returns structured challenge objects specifying clip URLs, metadata (duration, fps, resolution, sport, scenario), Element ID, and window ID.

#### Phase 3: Cryptographic Salt Derivation {#phase-3:-cryptographic-salt-derivation}

Each validator generates deterministic but unpredictable salts to prevent pre-computation attacks. The salt derivation uses a VRF or PRF with the validator's private key: `salt_seed = VRF(validator_sk, "sv1_salt" || manifest_hash || element_id || clip_id || challenge_seq)`, then deterministically samples offset and stride parameters. This ensures salts are unpredictable to miners but verifiable by third parties, each validator generates different salts for the same challenge, and salt generation is reproducible for audits.

#### Phase 4: Pseudo-Ground Truth Generation {#phase-4:-pseudo-ground-truth-generation}

Validators construct reference annotations locally using pinned PGT recipes. Large vision-language models (Qwen2.5-VL-72B-Instruct \[[24](#ref24)\], InternVL3-78B \[[25](#ref25)\]) process RGB frames and optical flow overlays to generate initial annotations. Multiple model outputs are merged using NMS and clustering for consistency. Quality gates enforce minimum confidence thresholds, temporal agreement constraints (IoU ≥ 0.7), and bounded retries. All PGT generation uses fixed random seeds and deterministic algorithms ensuring reproducibility.

#### Phase 5: Miner Inference and Submission {#phase-5:-miner-inference-and-submission}

Miners retrieve challenges, apply validator-specific salts, and execute Element implementations to generate predictions. The inference process must satisfy latency compliance (p95 bounds), output format compliance (Element-specific JSON schemas), comprehensive telemetry reporting (performance and resource utilisation), and cryptographic signing for authenticity.

#### Phase 6: Validator Scoring and Evaluation {#phase-6:-validator-scoring-and-evaluation}

Validators compare miner outputs against locally-generated PGT using multi-pillar metrics. Each evaluation pillar (IoU, count accuracy, temporal consistency) is computed independently and weighted per Manifest specifications. Submissions exceeding RTF thresholds receive zero scores regardless of accuracy. Clip-level scores aggregate across the window using EWMA with specified half-life parameters.

#### Phase 7: Shard Emission and Evidence Recording {#phase-7:-shard-emission-and-evidence-recording}

Validators emit cryptographically signed evaluation shards serving as permanent performance evidence. Each shard contains complete evaluation results and intermediate metrics, latency measurements and RTF calculations, telemetry data and resource utilisation statistics, cryptographic proofs of PGT recipe compliance, and validator signatures with attestation data.

#### Phase 8: Aggregation and Outlier Detection {#phase-8:-aggregation-and-outlier-detection}

Score's centralised aggregation service merges validator shards to produce final weight vectors. MAD analysis identifies and removes anomalous validator assessments weighted by stake. Raw scores are normalised across miners and adjusted for baseline requirements. Final emission weights are computed using difficulty factors (β) and baseline gates (θ).

#### Phase 9: Quality Assurance and Human Audit {#phase-9:-quality-assurance-and-human-audit}

A subset of challenges undergoes human annotation to validate PGT quality. Canary Gold hand-labeled challenges replace PGT for selected evaluations, anchoring ground truth. Disagreement sampling queues cases where miners consistently outperform PGT for human review, identifying potential PGT errors. Miners whose predictions are validated by human audit receive bonus multipliers (1.00-1.03x) in subsequent windows.

### 3.3 Network Participants and Responsibilities {#3.3-network-participants-and-responsibilities}

The Score Vision network comprises four distinct participant types, each fulfilling critical functions in the evaluation and reward distribution pipeline. Validators serve as the primary evaluation authority, constructing pseudo-ground truth annotations locally using pinned recipes, executing miner submissions against these references, and emitting cryptographically signed evidence shards that document all evaluation results. Their role is computationally intensive, requiring substantial resources to run large vision-language models for PGT generation while maintaining the determinism necessary for audit and reproducibility.

Aggregators, operated by Score in the current centralised governance model, perform the critical function of merging evaluation evidence from multiple validators into consensus weight vectors. This process involves statistical outlier detection using Median Absolute Deviation analysis weighted by validator stake, normalisation of raw scores across the miner population, and computation of temporally smoothed performance metrics. The aggregation service publishes final weight vectors both on-chain for settlement and in public indices for transparency and independent verification.

Miners represent the competitive core of the network, implementing Element and Agent capabilities that are continuously evaluated under real-world constraints. They must maintain online availability, satisfy strict latency requirements, adhere to standardised I/O contracts, and provide comprehensive telemetry to validate their performance claims. The miner population drives innovation through direct competition, with economic rewards flowing exclusively to those who demonstrate measurable improvements above established baselines.

Gold partners provide the human annotation layer that grounds the evaluation system in verified truth. Working with specialised annotation platforms such as Label Studio \[[27](#ref27)\] or CVAT \[[28](#ref28)\], these partners label carefully selected challenge subsets that serve as canary datasets replacing PGT for validation purposes. They also review disagreement cases where miner outputs systematically diverge from PGT, identifying potential errors in the automated annotation system and enabling discovery credits for miners who correctly detect PGT failures.

### 3.4 The Manifest: Cryptographic Rule Specification {#3.4-the-manifest:-cryptographic-rule-specification}

The Manifest serves as the cryptographically signed, content-addressed rulebook that governs each evaluation window, establishing a complete and immutable specification of all parameters, constraints, and procedures that validators and miners must follow. This design ensures that all network participants operate under identical rules while enabling transparent verification of compliance and deterministic reproduction of evaluation results.

Each Manifest specifies the complete set of challenge clips with their associated importance weights, enabling targeted evaluation of specific scenarios or edge cases deemed critical for current development priorities. The pre-processing pipeline is precisely defined, including frame sampling rates (canonical 5 FPS for evaluation), image resising parameters (long edge to 1280 pixels by default), and RGB normalisation procedures. This standardisation ensures that all participants evaluate models under identical input conditions, eliminating preprocessing as a source of variance in evaluation results.

Evaluation metrics are fully specified within the Manifest, including the weighted composition of multi-pillar scoring functions, threshold values for acceptance criteria, and any Element-specific evaluation logic. Latency constraints are expressed as p95 bounds in milliseconds, coupled with service rate declarations that establish the real-time viability requirements through the RTF framework. The pseudo-ground truth recipe is identified by its SHA-256 hash, cryptographically binding validators to specific PGT generation procedures without revealing the salting mechanisms that protect against pre-computation attacks.

Economic parameters embedded in the Manifest include Element-specific baseline thresholds (θ_e) below which no rewards are distributed, the delta floor ensuring meaningful improvement requirements, and difficulty weight multipliers (β_e) that tilt rewards toward more challenging capabilities. The Manifest declares its semantic version for compatibility tracking and specifies an expiry block number, after which the rules are no longer valid and a new Manifest must be published. Once signed and distributed through content-addressed storage, the Manifest becomes the authoritative reference for the evaluation window, with its hash serving as a compact verification mechanism in all protocol communications.

### 3.5 Challenge API and Distribution {#3.5-challenge-api-and-distribution}

The Challenge API provides miners with evaluation tasks through a simple REST endpoint: `GET /api/challenge → { clip_url, meta, element_id, window_id }`. Challenge cadence is defined in blocks within the Manifest (default 300 blocks), with real-time duration varying by chain block time. Requests include Manifest hash headers for verification and HMAC/EdDSA authentication with idempotent tokens. Error codes (401/404/409/410) handle authentication failures, missing windows, rate limits, and expired windows respectively. The protocol requires no changes to existing miner implementations.

### 3.6 Cryptographic Salting Mechanism {#3.6-cryptographic-salting-mechanism}

Each validator draws deterministic salts preventing pre-computation attacks. The interface uses VRF or PRF with validator secret keys: `salt_seed = VRF_or_PRF(sk, "sv1_salt" || manifest_hash || element_id || clip_id || challenge_seq)`, then deterministically samples offset ∈ {0,1,2,3,4} and stride ∈ {5,6}. The specific VRF algorithm (ed25519 or BLS) will be finalised with published test vectors before v1.3 lockdown. Shards carry validator public keys and proofs enabling third-party verification.

### 3.7 Real-Time Viability Framework {#3.7-real-time-viability-framework}

Real-time viability is defined against declared service rates rather than specific hardware. All Elements are evaluated at canonical 5 FPS for comparability, while service rate r_e specifies the frames-per-second the Element must support in production. The Real-Time Factor is calculated as RTF_e \= (t_p95_ms / 1000\) × (r_e / 5), with RTF_e ≤ 1.0 required to pass. Default service rates are r_player \= 25 FPS, r_ball \= 25 FPS, and r_calib \= 5 FPS, reflecting that calibration can run sub-second with interpolation between updates.

For Agents, the composed pipeline must satisfy RTF ≤ 1.0 relative to declared per-Element service rates and jitter ≤ 40ms, evidenced by telemetry. This avoids binding viability to specific hardware while maintaining honest real-time requirements.

### 3.8 Pseudo-Ground Truth Generation and Validation {#3.8-pseudo-ground-truth-generation-and-validation}

Pseudo-ground truth serves as the reference against which miner submissions are evaluated, constructed locally by each validator using cryptographically pinned recipes that ensure deterministic, reproducible annotation generation. This approach enables continuous automated evaluation at scale while maintaining evaluation quality through systematic validation against human annotations. The PGT generation process balances automation efficiency with quality assurance through multi-model ensembles, consensus mechanisms, and human audit integration.

The PGT construction pipeline begins with deterministic frame sampling, selecting a fixed number of frames per clip (default 3\) based on the validator-specific salt to ensure miners cannot predict which frames will be evaluated. Large vision-language models process these frames through multiple passes: initial annotation using models like Qwen2.5-VL-72B-Instruct applied to both RGB frames and optical flow overlays to capture motion information, followed by merging of multiple model outputs through Non-Maximum Suppression and clustering algorithms to produce consistent bounding boxes and tracking information. For sports applications, a secondary VLM pass using models like InternVL3-78B extracts palette and uniform features to assign team identifications and role classifications. All prompts are deterministic and outputs are JSON-schema validated to ensure structural consistency.

Quality gates enforce minimum standards throughout PGT generation. Confidence thresholds filter low-quality predictions, temporal agreement constraints require IoU ≥ 0.7 between consecutive frame annotations to ensure tracking consistency, and bounded retry mechanisms allow re-generation when initial attempts fail quality checks without enabling unbounded computation. The output produces complete annotations including bounding boxes, role classifications, team color assignments, and frame indices, all passed to the scoring subsystem for miner evaluation.

Human annotation provides critical grounding for the automated PGT system. Canary Gold—hand-labeled challenges replacing PGT for selected evaluations each window—anchors automated evaluation in verified truth and receives up-weighting during aggregation. Disagreement sampling identifies cases where miners consistently produce outputs diverging from PGT, queuing these for human review to detect potential PGT errors. When human review validates that miners correctly detected objects or events that PGT missed, discovery credits (multipliers G_i ∈ \[1.00, 1.03\]) reward these miners, creating incentives to find and report evaluation system failures. The comprehensive benchmarking methodology described in Section 4 establishes quantitative relationships between PGT quality, VLM baseline performance, and human annotation standards through systematic comparison and continuous monitoring.

### 3.9 Scoring and Aggregation Framework {#3.9-scoring-and-aggregation-framework}

The scoring and aggregation system transforms raw evaluation results into final emission weights through a multi-stage process combining validator-local computation, temporal smoothing, outlier detection, and centralised consensus formation. This architecture separates concerns between validators who perform detailed evaluation and the aggregation service that synthesises multiple validator perspectives into authoritative weight vectors.

Each validator performs off-chain scoring independently. Multi-pillar metrics are computed for each clip, with pillar scores weighted according to Manifest specifications to produce a composite clip metric. The latency gate is enforced strictly—any submission exceeding RTF thresholds receives a zero score regardless of accuracy metrics. Clip-level scores aggregate across the evaluation window into an Element score S*e using Exponentially Weighted Moving Averages with half-life h \= 3 windows, calculated as α \= 1 − 2^(−1/h) ≈ 0.2063, giving WindowScore_t \= α · ClipMean_t \+ (1 − α) · WindowScore*{t−1}. This EWMA formulation provides temporal smoothing that responds to recent performance while maintaining stability against single-window anomalies.

Improvement above baseline is calculated as Q_e \= max(S_e − θ_e, 0), ensuring that only performance exceeding Element-specific baseline thresholds θ_e earns rewards. The difficulty weight is applied as W_e \= β_e · Q_e, tilting rewards toward more challenging Elements. Discovery multipliers G_i ∈ \[1.00, 1.03\] are applied to miners whose predictions were validated by human audit as correctly identifying objects or events that PGT missed. Validators then prune outliers using stake-weighted Median Absolute Deviation analysis, removing anomalous assessments before producing their local weight vector documenting miner performance as observed from their perspective.

The centralised aggregation service operated by Score merges validator weight vectors into the final per-window weights. The aggregation process identifies and removes systematic outliers through cross-validator consistency checks, normalizes raw scores across the miner population to account for absolute performance level variations, and applies final baseline gates and difficulty weights to compute emission allocations. Critically, the protocol uses no commit-reveal mechanisms—all validator shards and the final weight vector are published openly, enabling anyone to independently recompute weights from public evidence and verify aggregation correctness.

Burn routing handles underperforming Elements by redirecting their emission allocations. If an Element's improvement Q_e equals zero (meaning no miner exceeded the baseline), that Element's emission mass routes to a designated burn UID within the mechanism's weight vector, implementing automatic burn of creator emissions for capabilities where the competitive population has not yet achieved baseline performance. This creates economic pressure to either improve Element performance or adjust baseline thresholds to reflect achievable capability levels.

### 3.10 Shard Schema (v1.3) {#3.10-shard-schema-(v1.3)}

```json
{
  "window_id": "2025-10-23",
  "validator": "ss58:...",
  "element_id": "PlayerDetect_v1@1.0",
  "lane": "public",
  "manifest_hash": "sha256:...",
  "pgt_recipe_hash": "sha256:...",
  "salt_id": "uint64",
  "metrics": {
    "iou_placement": 0.87,
    "count_accuracy": 0.93,
    "palette_symmetry": 0.81,
    "smoothness": 0.9,
    "role_consistency": 0.88
  },
  "composite_score": 0.88,
  "latency_pass": true,
  "p95_latency_ms": 178,
  "telemetry": {
    "gpu_mem_mb_peak": 4800,
    "jitter_ms": 32,
    "frames_egress": false
  },
  "tee_attestation": null,
  "signature": "nacl:..."
}
```

Trusted shards set `"lane": "trusted"` and include `tee_attestation { tee_type, measurement, container_digest, policy_id, report }` and `agent_integration_score`.

### 3.11 Human Audit and Quality Assurance {#3.11-human-audit-and-quality-assurance}

Human annotation provides the critical grounding layer that ensures automated evaluation remains aligned with genuine capability assessment. The human audit system operates through two complementary mechanisms: Canary Gold for proactive validation and disagreement sampling for reactive error detection. Together, these create continuous feedback loops that identify and correct systematic biases in automated evaluation while rewarding miners who discover evaluation failures.

Canary Gold constitutes a small but carefully selected subset of challenges per evaluation window that undergo professional human annotation using standardised schemas in platforms like Label Studio or CVAT. These hand-labeled challenges completely replace PGT for scoring purposes, providing ground truth anchors that prevent automated evaluation drift. The Canary Gold challenges are up-weighted during aggregation, giving them disproportionate influence on final scores relative to their numerical fraction of total challenges. This up-weighting ensures that evaluation remains calibrated to human judgment even as the automated PGT system processes the bulk of evaluations at scale. Canary selection targets edge cases, systematic challenge types where PGT accuracy is uncertain, and representative samples spanning the full difficulty distribution.

Disagreement sampling implements reactive quality control by identifying cases where miner outputs systematically diverge from PGT predictions. When multiple independent miners consistently detect objects or events that PGT missed, or when miners agree with each other but disagree with PGT on specific challenges, these discrepancies trigger human review. Annotators examine the flagged cases to determine whether miners correctly identified genuine objects that PGT failed to detect (true positives) or whether miners are producing false positives that coincidentally align across multiple implementations. Verified wins—cases where human review confirms miners were correct and PGT was wrong—add to the Gold set and earn discovery credits for the identifying miners. These discovery credits manifest as small multipliers G_i ∈ \[1.00, 1.03\] applied to the miner's rewards within the current window or carrying into the next window, creating economic incentives to identify and report evaluation system failures.

For the Trusted lane where privacy constraints may preclude human review of raw frames, synthetic canaries derived from redundant validator ensembles provide an alternative quality assurance mechanism. Multiple validators process challenges using different salt parameters and PGT recipes, with high agreement across validator outputs serving as a confidence signal approximating human validation. No raw frames or model weights leave the TEE enclave, preserving privacy while still enabling quality monitoring through cross-validator consensus metrics.

### 3.12 Network Architecture and Data Flow {#3.12-network-architecture-and-data-flow}

The protocol implements a pull-based networking model where miners actively fetch challenges rather than receiving pushed assignments, enabling better load distribution and miner autonomy. Miners retrieve video clips via CDN URLs specified in challenge responses, with content delivery optimised for global distribution and high availability. Validators maintain structured storage organising evaluation artifacts into distinct directories: `/responses/` stores miner predictions indexed by challenge and miner ID, `/evaluations/` contains signed shards documenting assessment results, and `/index.json` provides manifest and metadata enabling discovery and verification. All stored content is content-addressed via cryptographic hashing and digitally signed, enabling tamper detection and provenance verification.

### 3.13 Anti-Gaming Mechanisms and Security {#3.13-anti-gaming-mechanisms-and-security}

The protocol incorporates multiple layers of anti-gaming defenses addressing diverse attack vectors. Pre-computation attacks where miners attempt to cache solutions for known challenges are prevented by per-validator salting that ensures each validator presents unique challenge variants unpredictable until evaluation time. Slow but accurate models that achieve high scores while violating real-time requirements are zeroed by strict RTF latency gates that reject submissions exceeding service rate thresholds regardless of accuracy. Validator collusion attempts are detected and mitigated through stake-weighted MAD analysis that identifies and prunes outlier assessments, combined with cross-validator consistency requirements that make coordinated bias expensive to maintain.

Metric overfitting where miners optimise for specific evaluation quirks rather than genuine capability is countered through multi-pillar evaluation that prevents single-dimension gaming, Gold anchor validation that grounds automated metrics in human judgment, and periodic baseline rotation that prevents miners from memorising fixed performance targets. Noisy PGT that might contain systematic errors is corrected through the Canary Gold system providing human-validated ground truth and disagreement sampling identifying systematic PGT failures. Model weight leaks from the Trusted lane are prevented by TEE no-egress policies enforced through attestation verification, sealed storage protecting sensitive parameters, and container digest checks ensuring execution environment integrity.

### 3.14 Evaluation Window Lifecycle {#3.14-evaluation-window-lifecycle}

Each evaluation window progresses through a structured lifecycle defining when different protocol participants act and what evidence they produce. At T0, the signed Manifest is published establishing all rules for the upcoming window. At T1, validators request challenges and construct PGT locally using pinned recipes and validator-specific salts. At T2, miners retrieve challenges, execute inference, and submit predictions with telemetry. At T3, validators compute scores, enforce latency gates, and emit signed shards documenting evaluations. At T4, the aggregation service merges validator shards, applies outlier detection, produces final weight vectors, and pushes them on-chain for settlement. At T5, Gold sampling and human annotation occur for Canary challenges and disagreement cases. At T6, the next window begins with EWMA carry-forward of performance metrics, establishing continuity across windows.

### 3.15 Trusted Execution Environment Integration {#3.15-trusted-execution-environment-integration}

The Trusted lane initially targets two Bittensor subnets—Chutes and Targon—as the execution environment for privacy-preserving model evaluation. These subnets provide attestation capabilities that prove code integrity and enforce no-egress policies, enabling miners to compete while keeping model weights confidential. Attestation signals including TEE type, measurements or PCRs, container digests, and policy identifiers are integrated into shard metadata, enabling validators and aggregators to verify that evaluation occurred within properly configured secure enclaves. The no-egress policy restricts data exfiltration to structured outputs only—predictions and telemetry—preventing model weight extraction while still enabling performance verification.

As Chutes \[[22](#ref22)\] and Targon \[[23](#ref23)\] stabilize, the protocol may expand to support conventional TEE technologies including AWS Nitro Enclaves \[[15](#ref15)\] for cloud deployment, Intel SGX with DCAP attestation \[[16](#ref16)\] for general-purpose computing, AMD SEV-SNP \[[17](#ref17)\] for confidential VMs, and NVIDIA Confidential Compute \[[18](#ref18)\] for GPU-accelerated workloads. Each TEE technology will require equivalent attestation and policy verification adapted to its specific trust model and attestation format, but all implementations must satisfy the same functional requirements: cryptographic proof of code integrity, enforcement of no-egress policies, and protection of sensitive model parameters through sealed storage or memory encryption.

### 3.16 Reproducibility and Audit Framework {#3.16-reproducibility-and-audit-framework}

Complete reproducibility of evaluation results is a core protocol requirement enabling independent verification and dispute resolution. Any score can be recreated given three inputs: the Manifest specifying all rules and parameters, the shard index containing validator assessments and miner submissions, and Gold hashes identifying human-annotated ground truth. PGT recipes are pinned by cryptographic hash in the Manifest, binding validators to specific annotation procedures while enabling third parties to verify that validators followed declared recipes.

Manifest changes affecting evaluation outcomes require one full window's advance notice, giving miners time to adapt implementations to new requirements and preventing retroactive rule changes that would unfairly disadvantage participants. Emergency patches addressing security vulnerabilities or critical bugs may be deployed immediately but must be logged with detailed justification, enabling post-hoc audit of whether emergency procedures were appropriately invoked. This reproducibility framework maintains mechanism auditability even under centralised governance, ensuring that protocol operators cannot manipulate outcomes without leaving detectable evidence.

---

## 4\. Evaluation Methodology and Benchmarking Framework {#4.-evaluation-methodology-and-benchmarking-framework}

### 4.1 Multi-Tier Validation Architecture {#4.1-multi-tier-validation-architecture}

The Score Vision protocol implements a sophisticated three-tier validation system that establishes performance baselines, tracks competitive progress, and ensures that automated evaluation systems maintain alignment with human judgment. This framework addresses a fundamental challenge in decentralised AI evaluation: how to maintain evaluation quality while enabling continuous, automated assessment at scale.

### 4.2 State-of-the-Art VLM Baseline Establishment {#4.2-state-of-the-art-vlm-baseline-establishment}

For every new Element or Agent introduced to the protocol, we conduct comprehensive baseline benchmarking using state-of-the-art vision-language models (VLMs) operating without real-time constraints. This process establishes three critical reference points: the performance ceiling achievable with unlimited computational resources, the systematic biases and failure modes inherent in current automated evaluation approaches, and the target performance level that decentralised miners should aspire to match or exceed.

The baseline establishment process begins with the selection of multiple frontier VLMs representing different architectural approaches and training methodologies. For the initial deployment, this includes models such as Qwen2.5-VL-72B-Instruct \[[24](#ref24)\], InternVL3-78B \[[25](#ref25)\], and Moondream 3 Preview \[[26](#ref26)\]. These models are evaluated on carefully curated challenge sets that span the full range of difficulty and edge cases expected in production deployment. Importantly, these baseline evaluations are conducted without the latency constraints imposed on miners, allowing us to isolate pure capability assessment from real-time performance requirements.

Each baseline VLM processes challenges using multiple inference strategies: direct single-pass predictions, multi-step reasoning with chain-of-thought prompting, ensemble approaches that aggregate multiple model outputs, and iterative refinement where models review and improve their initial predictions. This comprehensive evaluation provides insight into the performance ceiling for each Element and reveals which aspects of the task benefit most from additional computation or reasoning steps. The results establish both accuracy benchmarks (what level of performance is technically achievable) and efficiency frontiers (how performance degrades as computational budgets tighten).

### 4.3 Human Annotation Gold Standard {#4.3-human-annotation-gold-standard}

Parallel to VLM baseline evaluation, professional human annotators label identical challenge sets using standardised annotation protocols and quality assurance procedures. This human annotation serves multiple critical functions: it provides ground truth for evaluating both VLM baselines and miner submissions, it reveals systematic biases where automated systems consistently diverge from human judgment, and it establishes inter-annotator agreement metrics that calibrate our expectations for achievable performance given inherent task ambiguity.

The human annotation process employs established tools such as Label Studio and CVAT, with annotators receiving detailed instructions that specify decision criteria for ambiguous cases. For object detection tasks, annotators mark bounding boxes, assign class labels, and indicate confidence levels based on visibility and occlusion. For keypoint detection and calibration tasks, annotators identify geometric features and validate spatial relationships against known constraints. Multiple independent annotators label each challenge, enabling computation of inter-annotator agreement metrics that quantify task difficulty and inherent ambiguity.

Quality control mechanisms include regular calibration sessions where annotators discuss challenging cases and refine shared understanding of annotation standards, statistical monitoring of annotator consistency and drift over time, and blind insertion of previously labeled challenges to detect annotator fatigue or systematic shifts in judgment. The resulting human annotations are version-controlled with cryptographic hashes, enabling reproducible evaluation even as annotation quality improves over time.

### 4.4 Continuous Miner Performance Tracking {#4.4-continuous-miner-performance-tracking}

With VLM baselines and human annotations established, the protocol tracks top miner performance continuously across evaluation windows. This tracking reveals the competitive progress of the decentralised miner population relative to both automated baselines and human-level performance. We publish comparative performance metrics that show, for each Element, the current performance distribution across miners, the gap between top miners and VLM baselines, and the alignment between miner outputs and human annotations versus PGT references.

The comparative framework measures performance along multiple dimensions simultaneously. Absolute accuracy metrics compare miner outputs directly against human annotations, revealing how close the decentralised network has come to matching human-level performance. Efficiency-adjusted metrics normalize performance by computational cost, showing which miners achieve the best accuracy-latency trade-offs and highlighting innovations in model optimization and architecture. Consistency metrics track temporal stability and agreement across different validators' evaluations, ensuring that competitive pressure drives genuine capability improvement rather than gaming of specific evaluation quirks.

A critical aspect of continuous tracking is the identification of systematic divergences between miner outputs, VLM baselines, and human annotations. When miners consistently produce predictions that diverge from VLM-generated PGT but align with human annotations, this signals potential systematic bias in the PGT generation process. Such cases trigger detailed investigation, with challenging examples sent for additional human review and potential updates to PGT generation procedures. Conversely, when top miners converge toward solutions that differ from both VLM baselines and human annotations, this may indicate either gaming behavior or genuine algorithmic innovation that the protocol should investigate carefully.

### 4.5 Rapid Convergence Incentives {#4.5-rapid-convergence-incentives}

The evaluation framework is explicitly designed to incentivize rapid convergence toward and eventual surpassing of VLM baseline performance, with the critical constraint that miners must satisfy strict real-time performance requirements while competing on accuracy. This creates a fundamentally different optimization problem than traditional ML research: miners cannot simply throw unlimited compute at problems but must develop genuinely efficient algorithms and architectures.

The difficulty weight mechanism (β factors) provides economic incentives that tilt rewards toward Elements where the performance gap relative to baselines remains large. As miners collectively approach VLM baseline performance on a particular Element, the protocol may adjust baseline thresholds upward, ensuring that rewards continue to flow only to participants pushing beyond previous performance ceilings. This creates a continuous improvement dynamic where the bar for reward eligibility rises with collective capability advancement.

Latency gates enforce the real-time viability constraint that distinguishes production-ready systems from research demonstrations. As miner populations develop more efficient solutions that approach VLM accuracy while meeting strict RTF requirements, the protocol validates that decentralised competition can solve the accuracy-efficiency trade-off that plagues production ML deployment. Published benchmarks explicitly track the Pareto frontier of accuracy versus latency, celebrating miners who achieve breakthrough improvements on either dimension without sacrificing the other.

### 4.6 Forward Reasoning and Predictive Elements {#4.6-forward-reasoning-and-predictive-elements}

A future evolution of the protocol will introduce a novel class of Elements focused on forward reasoning: the prediction of future events based on the temporal integration of outputs from lower-level perception Elements. These forward-reasoning Elements represent a qualitative expansion of capability beyond pure perception into anticipation and strategic understanding.

#### 4.6.1 Architectural Foundation {#4.6.1-architectural-foundation}

Forward-reasoning Elements consume time-series outputs from multiple perception Elements as their input. For example, a goal-scoring prediction Element might integrate Player positions and trajectories from PlayerDetect and PlayerXY Elements, ball dynamics from BallDetect and BallXY Elements, and geometric context from PitchCalib Elements. By reasoning over these integrated percepts, the Element can predict the probability of a goal being scored within the next N seconds and identify which players are most likely to be involved.

The technical architecture for forward reasoning differs significantly from pure perception Elements. While perception Elements process individual frames or short temporal windows with minimal state, forward-reasoning Elements maintain longer temporal context and build explicit world models. These models may be implemented through recurrent neural networks that accumulate evidence over time, transformer architectures that attend to relevant historical percepts, graph neural networks that model entity interactions and relationships, or hybrid neuro-symbolic systems that combine learned pattern recognition with explicit reasoning rules.

#### 4.6.2 Evaluation Methodology {#4.6.2-evaluation-methodology}

Evaluating forward-reasoning Elements poses unique challenges because predictions concern future events that have not yet occurred at inference time. The protocol addresses this through retrospective evaluation: miners make predictions at time t about events in the window \[t, t+Δ\], and these predictions are scored after the prediction window has elapsed and ground truth for event occurrence can be established. This requires maintaining prediction logs with cryptographic commitments to prevent retroactive modification, temporal alignment between predictions and eventual outcomes, and calibrated probability scoring that rewards both accuracy and appropriate confidence levels.

Evaluation metrics for forward reasoning emphasize prediction calibration and actionable lead time. Brier scores measure the accuracy of probabilistic predictions, penalising both over-confidence in incorrect predictions and under-confidence in correct predictions. Temporal precision metrics reward predictions that provide sufficient advance warning to enable reactive responses—a prediction made 100ms before an event has far less utility than one made 5 seconds prior. Conditional accuracy metrics assess how well predictions discriminate between different outcome scenarios, beyond simple binary event occurrence.

For sports applications like football, simulation environments provide controlled testbeds for training and evaluating forward-reasoning capabilities. Physics-based simulators can generate synthetic match scenarios with known ground truth for future events, enabling systematic evaluation of prediction accuracy across diverse tactical situations. These simulations serve dual purposes: miners can use simulated data to train predictive models before deploying to real-world evaluation, and validators can use simulation-based challenges to supplement real video evaluation, particularly for rare events (penalty kicks, corner kicks, counter-attacks) that occur infrequently in organic footage. Simulation fidelity is validated by comparing statistical distributions of events, player movements, and tactical patterns against professional match data, ensuring that simulation-trained models transfer effectively to real-world prediction tasks.

#### 4.6.3 Integration with Perception Elements {#4.6.3-integration-with-perception-elements}

The introduction of forward-reasoning Elements creates a hierarchical capability structure where higher-level reasoning depends on the reliability of lower-level perception. This dependency introduces novel challenges for protocol design: reward attribution becomes more complex when a forward-reasoning Element's failure might stem from errors in its perception inputs rather than its own reasoning logic, and the composability guarantees that enable independent Element development must be extended to handle temporal dependencies and cascading uncertainty.

The protocol addresses these challenges through explicit dependency declarations in Element contracts, telemetry requirements that track input quality and reasoning confidence separately, and uncertainty propagation mechanisms that enable forward-reasoning Elements to calibrate their confidence based on the reliability of their perception inputs. Economic incentives are structured to reward the entire dependency chain: when a forward-reasoning Element achieves strong performance, its constituent perception Elements receive bonus credits proportional to their contribution to the integrated capability.

#### 4.6.4 Phased Deployment Strategy {#4.6.4-phased-deployment-strategy}

Forward-reasoning Elements will be introduced through a carefully staged deployment process. Initial shadow deployment will run forward-reasoning evaluations in parallel with perception Elements without economic impact, validating evaluation methodology and building baseline performance datasets. Beta deployment will allocate a small fraction of emissions to experimental forward-reasoning Elements, enabling early miners to develop approaches while limiting risk. Full production deployment will occur only after the evaluation methodology has been validated through shadow operation and sufficient miner participation ensures competitive dynamics.

The initial forward-reasoning Elements will focus on relatively short prediction horizons (1-5 seconds) and well-defined events with clear ground truth. As the protocol and miner population mature, we will expand to longer prediction horizons, more complex event definitions, and strategic reasoning that requires modeling opponent intentions and multi-step lookahead. This progression will demonstrate how decentralised competition can drive not just perceptual capabilities but genuine reasoning and anticipation, moving the protocol toward comprehensive video understanding that rivals human cognitive abilities. Forward-reasoning Elements will follow the same rigorous benchmarking methodology established in Section 4, with VLM baselines providing upper bounds on predictive accuracy and human annotations establishing ground truth for retrospective evaluation of temporal predictions.

---

## 5\. Economic Model and Incentive Mechanisms {#5.-economic-model-and-incentive-mechanisms}

### 5.1 Design Principles and Objectives {#5.1-design-principles-and-objectives}

The Score Vision economic model aligns network incentives with practical deployment requirements through a carefully structured reward system. At its foundation, the protocol exclusively rewards measurable improvements in capability above established baselines. This performance-based approach ensures that emissions flow only to participants who contribute genuine value to the network, preventing dilution of rewards through low-quality submissions or gaming strategies that optimise metrics without advancing actual capability.

Central to this design is the enforcement of real-time viability constraints. All rewards are gated by strict latency requirements (RTF ≤ 1.0), ensuring that economic incentives drive solutions suitable for production deployment rather than pure accuracy optimisation. This constraint forces miners to balance model sophistication with inference speed, mirroring the trade-offs faced in real-world computer vision applications. Without this gate, the network would naturally gravitate toward computationally expensive models unsuitable for live video processing.

The allocation mechanism incorporates difficulty weighting to address the problem of scarce capabilities. Elements that are harder to implement or less commonly solved receive higher reward multipliers (β factors), creating stronger economic incentives for developing solutions to challenging problems. This prevents the network from converging exclusively on well-studied tasks while neglecting valuable but difficult capabilities. The difficulty weights are explicitly specified in the Manifest and adjusted through governance based on empirical scarcity and commercial demand signals.

Transparency and reproducibility form critical pillars of the economic design. All parameters—baseline thresholds, β factors, EWMA decay rates, concentration guards—are publicly specified and verifiable from published evidence. This enables independent audit of reward calculations and provides clear dispute resolution mechanisms when miners challenge evaluation results. The protocol's commitment to reproducibility extends to archiving all evaluation data, PGT recipes, and Manifest versions required to recreate historical scoring.

As the network transitions from pure research incentives to commercial revenue integration, the economic model employs conservative pricing mechanisms designed to protect token holders while enabling sustainable revenue generation. The "Goldilocks Redemption" framework (detailed in Section 5.4) introduces USD-backed token redemption only after revenue accumulation exceeds emission costs, ensuring that commercial integration strengthens rather than dilutes token value.

Finally, the model inherits its base monetary policy from the Dynamic TAO (dTAO) framework, maintaining consistency with the broader Bittensor ecosystem. Token supply management, emission schedules, halving cycles, and validator economics all follow dTAO specifications, while Score Vision focuses its design effort exclusively on intra-subnet allocation mechanisms. This separation ensures ecosystem-wide coherence while enabling specialised incentive structures optimised for computer vision development.

### 5.2 Mathematical Framework for Reward Distribution {#5.2-mathematical-framework-for-reward-distribution}

#### 5.2.1 Emission Surface Partitioning {#5.2.1-emission-surface-partitioning}

The total emission allocation E_total for each evaluation window is partitioned between two parallel mechanisms:

**Public Track Allocation**:

```
E_public = (1 - γ_trusted) × E_total
```

**Trusted Track Allocation**:

```
E_trusted = γ_trusted × E_total
```

Where γ_trusted ∈ \[0,1\] represents the fraction of emissions allocated to the privacy-preserving Trusted Agents Track. This parameter is dynamically adjustable through governance mechanisms to balance open development with commercial privacy requirements.

#### 5.2.2 Element-Level Reward Calculation {#5.2.2-element-level-reward-calculation}

For each Element e in the Public Track, the reward calculation follows a multi-stage process:

**Step 1: Temporal Score Aggregation**

```
S_e,t = α × ClipMean_e,t + (1 - α) × S_e,t-1
```

Where α \= 1 \- 2^(-1/h) represents the EWMA decay factor with half-life h (default h \= 3 windows).

**Step 2: Baseline Gate Application**

```
Q_e,t = max(S_e,t - θ_e, 0)
```

Where θ_e represents the Element-specific baseline threshold below which no rewards are distributed.

**Step 3: Difficulty Weight Application**

```
W_e,t = β_e × Q_e,t
```

Where β_e represents the Element-specific difficulty multiplier that tilts rewards toward more challenging capabilities.

**Step 4: Miner Share Calculation**

```
w_i,e,t = W_e,t × (performance_i,e,t / Σ_j performance_j,e,t)
```

Where performance_i,e,t represents miner i's normalised performance on Element e during window t.

#### 5.2.3 Trusted Track Composite Scoring {#5.2.3-trusted-track-composite-scoring}

The Trusted Agents Track uses a composite scoring mechanism that evaluates integrated system performance:

```
A_i,t = w_player × S_player,i,t + w_ball × S_ball,i,t + w_calib × S_calib,i,t + w_integration × I_i,t
```

With default weights:

- w_player \= 0.35 (Player detection performance)
- w_ball \= 0.35 (Ball detection performance)
- w_calib \= 0.20 (Calibration performance)
- w_integration \= 0.10 (Integration quality bonus)

Where I_i,t represents the integration quality score based on RTF compliance, jitter bounds, and ID continuity metrics.

### 5.3 Phase 1: Baseline Emissions and Burn Mechanisms {#5.3-phase-1:-baseline-emissions-and-burn-mechanisms}

#### 5.3.1 Latency Gate Enforcement {#5.3.1-latency-gate-enforcement}

The foundation of Phase 1 rewards rests on strict real-time performance gates that ensure all compensated solutions meet production deployment standards. Every reward calculation incorporates a binary latency check: if a miner's solution violates the RTF ≤ 1.0 constraint, their performance score is immediately set to zero regardless of accuracy metrics achieved. This hard gate prevents the common failure mode in AI competitions where participants optimise for leaderboard metrics while delivering solutions too slow for practical use.

The RTF calculation provides hardware-agnostic evaluation by normalising processing time against the declared service rate for each Element:

```
RTF_i,e,t = (t_p95,i,e,t / 1000) × (r_e / 5)
```

Here, t_p95,i,e,t represents the 95th percentile processing time (in milliseconds) for miner i on Element e during window t, while r_e specifies the target service rate (frames per second) declared in the Element contract. Using the 95th percentile rather than mean or median ensures that solutions maintain consistent performance even under adverse conditions—a critical requirement for live video applications where occasional slowdowns would cause frame drops and degraded user experience.

The choice of p95 over stricter percentiles (p99, p99.9) balances robustness against measurement noise, while the 5 fps normalisation factor in the denominator reflects the typical frame sampling rate used during evaluation. A miner processing a 25 fps Element at 200ms p95 latency would compute RTF \= (200/1000) × (25/5) \= 1.0, exactly meeting the threshold. This framework enables fair comparison between miners using different hardware while maintaining the protocol's commitment to real-time viability.

#### 5.3.2 Concentration Guards and Anti-Monopoly Mechanisms {#5.3.2-concentration-guards-and-anti-monopoly-mechanisms}

Decentralised networks face a persistent challenge: without intervention, reward distributions often converge toward winner-take-all dynamics where a small number of high-performing participants capture the majority of emissions. While this reflects legitimate performance differences, excessive concentration can discourage new entrants and reduce network resilience. Score Vision addresses this through optional concentration guards that can be activated via governance when reward distribution becomes unhealthy.

The concentration mechanism applies a progressive penalty to reward shares exceeding a specified threshold:

```
s'_i,t = {
  s_i,t                           if s_i,t ≤ τ
  τ + (s_i,t - τ) × (1 - λ)      if s_i,t > τ
}
```

Here, s_i,t represents miner i's provisional reward share in window t, τ defines the concentration threshold (typically set around 30%), and λ ∈ \[0,1\] determines the burn rate applied to excess concentration. The default configuration (λ \= 1\) implements full burn of rewards above the threshold, creating a soft cap that preserves incentives for high performance while preventing monopolistic concentration. After applying these guards, shares are renormalised (s''\_i,t \= s'\_i,t / Σ_j s'\_j,t) to maintain total emission consistency.

The concentration threshold and burn rate are explicitly specified in the Manifest and can be adjusted through governance based on network health metrics. Conservative initial settings (τ \= 30%, λ \= 1\) provide strong anti-monopoly protection, but these parameters can be relaxed as the network matures and competition intensifies. Importantly, concentration guards apply per-window rather than cumulatively, ensuring that temporarily dominant miners don't face permanent penalties if new competitors emerge.

#### 5.3.3 Discovery Credit Mechanisms {#5.3.3-discovery-credit-mechanisms}

The protocol's reliance on Pseudo-Ground Truth creates an inherent tension: PGT must be high-quality enough to drive meaningful improvement, yet no automated system achieves perfect accuracy. Rather than treating PGT as infallible, Score Vision incorporates an economic mechanism that rewards miners for identifying errors through the human audit validation process. This creates productive feedback loops where miners are incentivised to challenge suspicious ground truth rather than merely conforming to potentially flawed labels.

When a miner's prediction differs significantly from PGT and human auditors subsequently validate the miner's answer as correct, the protocol applies a discovery credit multiplier to that miner's rewards for the affected window:

```
final_reward_i,t = G_i,t × base_reward_i,t
```

The multiplier G_i,t ∈ \[1.00, 1.03\] scales with the number and significance of validated corrections, providing up to a 3% reward boost for miners who successfully identify PGT errors. This mechanism serves multiple purposes: it compensates miners for potential score penalties caused by incorrect ground truth, creates economic incentives for reporting suspicions rather than gaming around known PGT weaknesses, and provides valuable signal to validators about which PGT generation strategies produce unreliable labels.

Discovery credits apply retroactively once human audit completes, typically within the T+1 to T+2 window after initial scoring. The modest 3% cap prevents excessive reward volatility while remaining meaningful enough to incentivise participation in the error-reporting process. Critically, only validated corrections earn credits—frivolous challenges that human auditors reject receive no bonus, preventing spam attacks on the audit system.

### 5.4 Phase 2: Commercial Revenue Integration ("Goldilocks Redemption") {#5.4-phase-2:-commercial-revenue-integration-("goldilocks-redemption")}

#### 5.4.1 Revenue-Backed Token Redemption {#5.4.1-revenue-backed-token-redemption}

Phase 2 introduces the "Goldilocks Redemption" mechanism, creating a sustainable bridge between commercial revenue and token economics. As Score Cloud and partner applications generate USD revenue from production deployments, a portion of this revenue flows into Element-specific redemption pools where token holders can exchange their emissions for USD backing. This mechanism addresses a critical challenge in decentralised AI networks: aligning the incentives of early contributors (who receive emissions) with the commercial success of the platform they helped build.

The redemption framework begins with careful attribution of commercial revenue to the specific Elements that generated it. For each Element e over evaluation horizon H (typically measured in months or quarters), the protocol tracks cumulative revenue:

```
Υ_e,H = Σ_{t=t_0}^{t_0+H} revenue_e,t
```

This attribution enables Element-specific redemption pricing, ensuring that miners who focused on commercially valuable capabilities benefit proportionally from that value creation. Simultaneously, the protocol tracks total token emissions for each Element over the same period:

```
Ξ_e,H = Σ_{t=t_0}^{t_0+H} emissions_e,t
```

The redemption price for Element e combines these metrics with difficulty weighting and prudence factors:

```
p_e,t = κ × (Υ_e,H / Ξ_e,H) × β_e × smoothing_factor_t
```

Here, κ ∈ (0,1\] acts as a prudence factor protecting against revenue volatility—typical values (κ \= 0.7-0.9) ensure that redemption prices remain conservative even if commercial revenue proves volatile or temporary. The β_e factor applies difficulty-adjusted pricing, allowing scarcer capabilities to command premium redemption rates even if absolute revenue volumes remain modest. Finally, smoothing_factor_t applies EWMA smoothing to prevent sharp price discontinuities that could destabilise the redemption market.

#### 5.4.2 Redemption Pool Management {#5.4.2-redemption-pool-management}

The redemption system must handle scenarios where token holder demand for USD redemption exceeds available reserves. Rather than implementing first-come-first-served mechanics that would create rushing dynamics and favor high-frequency traders, Score Vision employs pro-rata fulfillment that treats all redemption requests within a window equally:

```
fill_ratio_t = min(1, available_USD_t / total_requests_t)
```

When available USD reserves fall short of total requests, each participant receives a proportional fill of their request. This approach eliminates timing advantages and ensures fair access to liquidity across all token holders regardless of technical sophistication.

Unfilled requests don't simply disappear—they roll forward into subsequent windows with priority weighting:

```
priority_i,t+1 = ρ × priority_i,t + (1 - ρ) × unfilled_amount_i,t
```

The persistence parameter ρ ∈ \[0,1\] controls how aggressively the protocol prioritises backlog fulfillment. Conservative settings (ρ \= 0.3-0.5) give moderate priority to previously unfilled requests, while aggressive settings (ρ \= 0.1-0.2) ensure rapid backlog clearing once revenue inflows increase. This mechanism prevents indefinite backlog accumulation while avoiding scenarios where early requestors permanently crowd out new redemption attempts.

Finally, redemption prices are subject to explicit bounds that prevent manipulation or extreme volatility:

```
p_e,t^bounded = clamp(p_e,t, p_min, p_max)
```

These bounds protect against both predatory pricing (where redemption rates fall below sustainable levels) and speculative bubbles (where temporary revenue spikes create unsustainable redemption expectations). The bound parameters are explicitly specified in the economic Manifest and adjusted through governance as commercial traction establishes baseline revenue patterns.

#### 5.4.3 Transparency and Reporting {#5.4.3-transparency-and-reporting}

Effective redemption markets require comprehensive transparency about pool mechanics, pricing dynamics, and fulfillment rates. Score Vision publishes detailed economic data for each evaluation window, creating information symmetry between protocol operators and token holders. This reporting includes pool inflow (R_t, representing USD revenue), Element-specific redemption prices (p_e,t for each active Element), fill ratios (fill_ratio_t indicating what percentage of requests were fulfilled), backlog status (unfilled_requests_t tracking accumulated unmet demand), and complete emission statistics showing total and per-Element token distribution.

This transparency serves multiple critical functions. Token holders can make informed decisions about redemption timing—understanding whether current fill ratios justify immediate redemption or whether waiting for improved liquidity makes strategic sense. Miners receive clear market signals about which Elements generate meaningful commercial revenue, informing their development priorities and specialisation choices. The broader ecosystem gains visibility into the protocol's commercial traction and sustainability trajectory, enabling more accurate valuation and long-term planning.

All economic data is published on-chain or through cryptographically verifiable off-chain repositories, ensuring that no centralised operator can manipulate reporting to favor specific participants. Historical archives enable retrospective analysis of redemption market evolution, providing empirical data for future governance decisions about prudence factors, price bounds, and backlog priorities.

### 5.5 Integration with Dynamic TAO Framework {#5.5-integration-with-dynamic-tao-framework}

Score Vision operates as a specialised subnet within the broader Bittensor ecosystem, inheriting its base monetary policy from the Dynamic TAO (dTAO) framework while focusing design effort on domain-specific allocation mechanisms. This integration ensures ecosystem-wide consistency in fundamental economic parameters while enabling the flexibility required for computer vision's unique requirements.

The protocol adopts dTAO specifications for total token supply management, ensuring that Score Vision emissions align with network-wide supply schedules and halving cycles. This prevents inflation mismatches that could distort cross-subnet economic interactions or create arbitrage opportunities harmful to the broader ecosystem. Similarly, base emission rates follow dTAO parameters, with Score Vision's allocation decisions operating within the emission budget assigned by the parent chain.

Validator economics integrate tightly with dTAO's validator incentive framework. Validators participating in Score Vision evaluation earn rewards according to dTAO's staking and performance mechanisms, ensuring that validator behavior remains consistent across subnets and preventing fragmentation of the validator set. This integration is critical for maintaining Bittensor's security model, where validators must allocate attention across multiple subnets without creating exploitable inconsistencies.

Cross-subnet economic interactions—including token transfers, staking delegation, and economic signaling—follow dTAO protocols precisely. This enables Score Vision to participate in ecosystem-wide governance, liquidity pools, and coordination mechanisms without requiring special-case handling. The result is a subnet that benefits from Bittensor's established monetary infrastructure while focusing its specialised design on computer vision allocation mechanisms: difficulty weighting, RTF gates, Element-specific pricing, and redemption pool management.

This separation of concerns—dTAO for base monetary policy, Score Vision for allocation specifics—creates clear boundaries between ecosystem-wide consistency requirements and domain-specific optimisation opportunities. Future changes to dTAO parameters automatically propagate to Score Vision, while Score Vision's governance can independently adjust allocation mechanisms without requiring ecosystem-wide coordination.

---

## 6\. Commercialisation Strategy {#6.-commercialisation-strategy}

The transition from decentralised research network to commercially viable platform requires careful bridging between protocol-layer incentives and product-layer value delivery. Score Vision's commercialisation strategy operates on two tiers: Score Cloud provides infrastructure-level access to network capabilities, while Product Layer+ demonstrates vertical-specific applications that leverage this infrastructure. This separation enables both direct API monetisation and ecosystem development around specialised use cases.

### 6.1 Score Cloud: Unified Platform with Dual Execution Modes {#6.1-score-cloud:-unified-platform-with-dual-execution-modes}

Score Cloud serves as the primary commercial interface to the Score Vision network, providing a unified API and dashboard that converts raw video streams into structured data outputs. Customers upload video clips or connect live streams, specify their analysis requirements using Element and Agent configurations, and receive JSON or CSV outputs containing detections, tracks, coordinate transformations, and event classifications. All transactions include comprehensive telemetry covering latency profiles, accuracy metrics, and resource utilisation, enabling customers to validate that commercial deployments meet their specific performance requirements.

The platform operates in two distinct execution modes, each optimised for different customer needs and privacy constraints. Public Mode routes workloads through the transparent Public Track, where miners compete openly and all evaluation data contributes to protocol improvement. This mode offers cost-efficient processing suitable for non-sensitive applications: sports analytics using publicly broadcast footage, traffic monitoring with anonymised streams, or retail analytics where privacy regulations permit open processing. Public Mode pricing follows straightforward per-minute or per-frame billing, with rates determined by Element complexity and service rate requirements.

Trusted Mode addresses the substantial market segment requiring strict privacy guarantees. Customer video streams are routed exclusively to Trusted Track Agents operating within attested TEEs, ensuring that raw frames never leave the secure enclave and model weights remain confidential throughout execution. This mode targets applications like security surveillance (where footage contains sensitive personal data), proprietary manufacturing processes (where video reveals trade secrets), or medical imaging (covered by HIPAA and similar regulations). Trusted Mode commands a premium over Public Mode, reflecting both the additional infrastructure cost of TEE deployment and the economic value of privacy-preserving execution.

The operational flow integrates seamlessly across both modes. Customers upload video clips or configure streaming endpoints, select their execution mode based on privacy requirements, and specify which Elements or Agents should process their content. The platform automatically routes requests to the appropriate lane—Public Track for transparent workloads, Trusted Track for privacy-preserving execution—performs the analysis, and returns structured outputs with latency/accuracy telemetry. All transactions are logged for billing accuracy and quality assurance, creating audit trails that support both customer invoice disputes and protocol performance validation.

Pricing models balance simplicity with flexibility. Base pricing follows per-minute or per-frame structures, with rates varying by Element complexity, declared service rates, and execution mode. Trusted Mode adds explicit TEE surcharges reflecting the infrastructure premium for confidential computing. Enterprise customers can access optional enhancements including regional routing (for latency optimisation or data residency compliance), guaranteed SLAs with uptime commitments, priority processing during peak demand, and dedicated support channels for integration assistance and troubleshooting.

### 6.2 Product Layer and Vertical Applications {#6.2-product-layer-and-vertical-applications}

While Score Cloud provides general-purpose infrastructure, many customers require vertical-specific solutions with domain-tailored interfaces and workflows. Product Layer+ develops these specialised applications in collaboration with domain experts, demonstrating Score Vision's utility in concrete use cases while creating additional revenue streams. Initial pilots focus on areas where Score Vision's capabilities offer clear advantages: sports analytics dashboards providing real-time player tracking and tactical visualisations, broadcast overlay systems that augment live video with computer vision insights, industrial monitoring platforms for manufacturing quality control, and traffic analytics solutions for urban planning and congestion management.

These pilot applications serve dual purposes. First, they validate that Score Vision's Element-based architecture and real-time performance constraints actually deliver value in production deployments, providing empirical feedback that guides protocol development. Second, they create reference implementations that third-party developers can study when building their own Score Vision-powered applications, accelerating ecosystem growth.

The business model for Product Layer+ emphasises ecosystem enablement over centralised control. Pilot applications initially operate as co-built ventures, with Score Vision providing technical integration and domain partners contributing specialised knowledge and customer relationships. As applications stabilise and achieve product-market fit, they transition to independent operation with usage-based billing for Score Cloud API consumption. This graduation model prevents Score Vision from becoming a bottleneck to vertical market innovation while ensuring that protocol development benefits from commercial feedback.

Certified applications that meet quality and integration standards can carry "Powered by Score Vision" branding, signaling to customers that the application leverages the protocol's decentralised AI infrastructure. This certification program creates network effects: high-quality vertical applications attract customers to the Score Vision ecosystem, increasing commercial demand for Elements and driving protocol emissions toward practically valuable capabilities. The result is a virtuous cycle where commercial success reinforces protocol development rather than diverging from it.

---

## 7\. Security Analysis and Threat Model {#7.-security-analysis-and-threat-model}

### 7.1 Threat Landscape Overview {#7.1-threat-landscape-overview}

The Score Vision protocol operates in an adversarial environment where rational economic actors may attempt to maximise their rewards through means other than legitimate performance improvements. The security model must address both traditional distributed systems threats and novel attack vectors specific to decentralised AI evaluation networks.

### 7.2 Formal Threat Model {#7.2-formal-threat-model}

#### 7.2.1 Adversary Capabilities and Objectives {#7.2.1-adversary-capabilities-and-objectives}

We model adversaries with substantial computational resources bounded by economic constraints, capable of participating as miners or validators across multiple identities (Sybil attacks), with access to all public protocol information and network traffic patterns. We assume adversaries are computationally bounded and cannot break standard cryptographic primitives (SHA-256, Ed25519) within practical time frames. Their primary objectives include extracting emissions without legitimate value, biasing evaluation results through market manipulation, extracting private model weights or training data, and degrading network reliability or evaluation quality.

#### 7.2.2 Attack Surface Analysis {#7.2.2-attack-surface-analysis}

The protocol's attack surface spans three layers. Protocol layer attacks include challenge pre-computation and caching, evaluation metric gaming and overfitting, latency gate circumvention, and shard manipulation or forgery. Economic layer threats encompass validator collusion and vote buying, concentration attacks and market cornering, redemption pool manipulation, and cross-subnet arbitrage exploitation. Infrastructure layer vulnerabilities involve TEE side-channel exploitation, network-level traffic analysis, storage and CDN manipulation, and timing or covert channel attacks.

### 7.3 Specific Attack Vectors and Mitigations {#7.3-specific-attack-vectors-and-mitigations}

#### 7.3.1 Pre-Computation and Caching Attacks {#7.3.1-pre-computation-and-caching-attacks}

Miners might attempt to pre-compute responses to known challenges or cache solutions, avoiding real-time inference costs. The protocol mitigates this through per-validator cryptographic salting:

```
salt_seed = VRF(validator_sk, domain_separator || manifest_hash || challenge_id)
challenge_variant = apply_salt(base_challenge, salt_seed)
```

The VRF ensures salts remain unpredictable until challenge time, with a salt space exceeding 2^40 variants per challenge to prevent exhaustive pre-computation.

#### 7.3.2 Latency Gate Circumvention {#7.3.2-latency-gate-circumvention}

Miners might manipulate timing measurements through specialised hardware, system clock manipulation, or batching/caching schemes. The RTF framework provides hardware-agnostic evaluation based on declared service rates rather than absolute timing, with multiple independent timing measurements and outlier detection. Comprehensive telemetry verification requires detailed resource utilisation reports, making timing manipulation detectable through inconsistencies.

#### 7.3.3 Validator Collusion and Bias Injection {#7.3.3-validator-collusion-and-bias-injection}

Coordinated validators might provide biased scores or selectively apply different standards to favor specific miners. The protocol employs Median Absolute Deviation (MAD) analysis with stake weighting for robust outlier detection:

```
outlier_threshold = MAD_factor × median_absolute_deviation(validator_scores)
valid_scores = filter_outliers(all_scores, outlier_threshold, validator_stakes)
final_score = stake_weighted_average(valid_scores)
```

The public shard system enables post-hoc detection of systematic bias, while cross-validator consistency requirements make coordinated manipulation expensive and detectable.

#### 7.3.4 Pseudo-Ground Truth Poisoning {#7.3.4-pseudo-ground-truth-poisoning}

Adversaries might manipulate PGT generation through biased training data, prompt engineering, or infrastructure compromise. Mitigations include pinned PGT recipes with cryptographic hashes, multi-model ensembles with diverse architectures to reduce single-model bias, Canary Gold human-validated anchors, and disagreement sampling for systematic error detection. Cryptographic pinning prevents recipe manipulation, while statistical analysis of human-PGT agreement enables bias detection.

#### 7.3.5 TEE-Specific Attack Vectors {#7.3.5-tee-specific-attack-vectors}

Adversaries might exploit TEE vulnerabilities through side-channel attacks (timing, power, electromagnetic), memory access pattern analysis, covert channels, or implementation flaws. Multi-layer security combines comprehensive attestation verification (measurement and policy checks), no-egress policy enforcement with cryptographic verification, sealed storage for sensitive parameters, and regular security audits. The combination of multiple TEE technologies (Chutes, Targon, traditional TEEs) provides defense in depth with cryptographic proof of policy compliance.

#### 7.3.6 Economic Manipulation Attacks {#7.3.6-economic-manipulation-attacks}

Adversaries might attempt market cornering, redemption pool manipulation, cross-temporal arbitrage, or Sybil attacks to extract unfair value. Economic security employs concentration guards with configurable burn rates, EWMA smoothing for price stability, pro-rata redemption fulfillment, and stake-based Sybil resistance inherited from Bittensor's identity system. These complementary mechanisms prevent market cornering while reducing manipulation profitability.

### 7.4 Residual Risk Assessment {#7.4-residual-risk-assessment}

Despite comprehensive mitigations, residual risks remain: PGT generation may contain systematic biases favoring certain architectures or training methodologies; external market conditions may create arbitrage opportunities distorting network incentives; software bugs or configuration errors may introduce temporary attack vectors; and centralised parameter control creates single points of failure for protocol manipulation. All residual risks are continuously monitored through automated detection systems and regular security audits, with risk assessments updated per evaluation window and emergency response procedures maintained for critical vulnerabilities.

### 7.5 Security Verification and Audit Framework {#7.5-security-verification-and-audit-framework}

The protocol maintains continuous monitoring through automated systems tracking statistical anomalies in validator behavior, miner performance patterns, and economic metrics. Quarterly security audits by independent firms assess protocol implementation, TEE configurations, and economic mechanism integrity. An ongoing bug bounty program incentivises security researchers to identify and report vulnerabilities. Documented incident response procedures handle security incidents through emergency parameter updates and temporary mitigation deployment. Regular transparency reporting publishes security metrics, incident reports, and risk assessments to maintain community trust and enable independent verification.

---

## 8\. Governance & Upgrades {#8.-governance-&-upgrades}

### 8.1 Current Governance Model (v1.3) {#8.1-current-governance-model-(v1.3)}

The Score Vision protocol currently operates under centralised governance to ensure rapid iteration, parameter tuning, and course correction during the network's formative stages. Score, as subnet owner, maintains authority over critical protocol parameters including the Element set, difficulty weights (β), baseline thresholds (θ) and floors (δ), latency gates, service rates, trusted lane allocation (γ_trusted), PGT recipe specifications, telemetry field requirements, and API contract definitions.

This centralised approach enables the protocol to adapt quickly to empirical findings, address unforeseen gaming strategies, and optimise economic parameters based on real-world performance data. Without this flexibility, the network would risk lock-in to suboptimal parameters during its most critical developmental phase.

### 8.2 Change Management Process {#8.2-change-management-process}

Parameter changes follow structured processes that balance flexibility with predictability. Standard changes—including Element additions, baseline adjustments, and PGT recipe updates—are published with at least one evaluation window advance notice, shipped via updated Manifest specifications. This notice period allows miners and validators to prepare infrastructure, adjust implementations, and provide feedback before changes take effect.

Emergency changes address security vulnerabilities, fraud detection, or critical operational issues requiring immediate intervention. These hot-patches deploy with immediate notice but are accompanied by comprehensive post-mortem documentation published in the subsequent evaluation window. Emergency procedures are reserved for scenarios where delayed action would compromise network integrity or participant assets.

All parameter changes, whether standard or emergency, are logged in the public parameter index with detailed diffs showing before/after values and explicit rationale explaining the motivation for each change. This transparency enables the community to understand governance decisions, identify patterns in parameter evolution, and provide informed feedback on protocol development.

### 8.3 Path Toward Decentralisation {#8.3-path-toward-decentralisation}

While v1.3 operates under centralised governance, the long-term vision includes progressive decentralisation as the protocol matures and stabilises. Score will steward the network through its initial development, ensuring the protocol moves in the right direction and achieves product-market fit. As parameter stability increases and governance patterns become established, decision-making authority will gradually transition to community-driven mechanisms.

Future decentralisation may include community voting on Element additions, stake-weighted parameter adjustment proposals, elected governance committees for PGT recipe management, and transparent escalation procedures for disputes. The timing and structure of this transition will be determined by protocol maturity, community readiness, and the establishment of robust mechanisms that prevent governance capture or manipulation.

Any transition toward decentralised governance will be documented explicitly through formal proposals and community review processes. There is no fixed timeline for this transition; it will occur when empirical evidence demonstrates that decentralised mechanisms can maintain protocol integrity without sacrificing the adaptability required for continuous improvement.

---

## 9\. Implementation Details {#9.-implementation-details}

This section provides concrete implementation specifications for Elements, Agents, and protocol APIs. Full machine-readable JSON schemas and SDK documentation are maintained separately; these excerpts illustrate the core interfaces that enable interoperability between miners, validators, and the broader ecosystem.

### 9.1 Element and Agent I/O Schemas {#9.1-element-and-agent-i/o-schemas}

Elements and Agents communicate through standardised JSON schemas that ensure consistent interpretation of inputs and outputs across the network. The FrameBatch schema specifies video input with CDN-hosted frame URLs, frame rate metadata, and timing information:

```json
{
  "clip_id": "sha256:...",
  "fps": 5,
  "frames": ["https://.../f0001.jpg", "..."],
  "ts_start_ms": 1234567890
}
```

Object detection Elements (PlayerDetect, BallDetect) output Detections arrays with bounding boxes, confidence scores, tracking IDs, and optional semantic attributes:

```json
{
  "frame_idx": 12,
  "class": "player",
  "bbox": [x0, y0, x1, y1],
  "score": 0.93,
  "track_id": "t-184",
  "team_id": 0,
  "role": "goalkeeper"
}
```

Calibration Elements (PitchCalib) output keypoint annotations and homography matrices for coordinate transformation:

```json
{
  "frame_idx": 8,
  "points": [{"id":"corner_fl","x":123.4,"y":567.8}, ...],
  "homography": [[h00,h01,h02],[h10,h11,h12],[h20,h21,h22]]
}
```

Field coordinate outputs (XY arrays) represent transformed 2D positions in real-world units:

```json
{
  "frame_idx": 15,
  "track_id": "t-184",
  "xy": [23.41, 12.08],
  "unit": "meters"
}
```

### 9.2 Telemetry Requirements {#9.2-telemetry-requirements}

All Element and Agent submissions must include comprehensive telemetry enabling RTF calculation, resource verification, and gaming detection:

```
timing: {p50_ms, p95_ms, max_ms, jitter_ms}
throughput: {frames_in, frames_proc, service_rate_fps}
encoder: {reuse_ratio, batches}
continuity: {id_stability, id_switch_rate, drop_rate}
resources: {gpu_mem_mb_peak, cpu_pct_peak}
egress: {frames_egress:false}
trusted?: {tee_type, measurement, container_digest, policy_id}
```

Timing metrics support latency gate enforcement, throughput fields verify declared service rates, encoder statistics detect caching, continuity metrics assess tracking quality, resource measurements enable anomaly detection, egress flags enforce no-frame-export policies, and optional TEE attestation proves privacy compliance.

### 9.3 Challenge API Specification {#9.3-challenge-api-specification}

Miners retrieve evaluation challenges through authenticated API endpoints with cryptographic verification:

```
GET /api/challenge
Headers:
  X-Manifest-Hash: sha256:...
  X-Client-Id: uid
  X-Nonce: uuid
  X-Signature: ed25519(sig(base64url(payload)))
Errors:
  401 (invalid signature), 404 (no active window),
  409 (rate limit), 410 (expired window)
```

The Manifest hash ensures miners and validators agree on protocol parameters, client IDs enable rate limiting and tracking, nonces prevent replay attacks, and Ed25519 signatures authenticate requests. Detailed error codes enable clients to diagnose and handle failure scenarios appropriately.

---

## 10\. Roadmap {#10.-roadmap}

The Score Vision roadmap prioritises capabilities that expand practical utility while strengthening protocol robustness. Near-term development focuses on completing the initial sports analytics Element set, enhancing evaluation quality, and stabilising commercial integrations. Medium-term goals include multi-sport and industrial domain expansion, while long-term directions explore advanced TEE integration and economic model maturation.

### 10.1 Element Expansion {#10.1-element-expansion}

The immediate Element roadmap completes the football analytics suite with PlayerXY_v1 and BallXY_v1 for real-world coordinate tracking, and GoalSimple_v0.1 for event detection. These Elements enable end-to-end sports analytics workflows from detection through coordinate transformation to event classification. Following football stabilisation, cricket-specific Elements will address the second major sports vertical, demonstrating Element architecture generalisability across distinct visual domains. Industrial monitoring Elements targeting manufacturing quality control and safety compliance represent the protocol's expansion beyond entertainment applications into high-value enterprise markets.

### 10.2 Evaluation Infrastructure Enhancement {#10.2-evaluation-infrastructure-enhancement}

PGT generation will expand to multi-model ensembles incorporating additional frontier VLMs, reducing single-model bias and improving evaluation robustness. Stronger geometric consistency checks will catch physically impossible predictions, while published PGT-to-Gold benchmarks will provide transparent quality metrics enabling community assessment of ground truth reliability. These enhancements address current limitations in automated evaluation while maintaining the scalability advantages of PGT-based assessment.

### 10.3 Trusted Execution Environment Maturity {#10.3-trusted-execution-environment-maturity}

The Trusted Track will stabilize its initial implementation using Chutes (SN64) and Targon (SN4) subnets, establishing operational procedures and gathering empirical performance data. Following stabilization, the protocol may integrate conventional TEE technologies including AWS Nitro Enclaves, Intel SGX, AMD SEV-SNP, and NVIDIA Confidential Computing. This diversification provides defense in depth, reduces dependency on any single TEE vendor, and expands the pool of miners capable of participating in privacy-preserving execution.

### 10.4 Economic Model Activation {#10.4-economic-model-activation}

Phase 2 commercial integration will activate after the first external revenue cycles demonstrate sustainable demand and establish baseline revenue patterns. The redemption pool mechanism requires empirical calibration of prudence factors, price bounds, and backlog priorities—parameters that can only be properly tuned with real commercial data. Conservative activation ensures that commercial integration strengthens rather than destabilizes token economics.

### 10.5 Operational Transparency {#10.5-operational-transparency}

Public dashboards will expose key network health metrics including latency distributions (enabling performance trend analysis), Gold standard drift (tracking PGT quality evolution), discovery credit allocation (showing error-detection incentive effectiveness), and Element-specific performance trajectories. These dashboards enable participants to make informed decisions about infrastructure investment, development priorities, and redemption timing while demonstrating protocol health to external stakeholders.

---

## 11\. Conclusion and Future Directions {#11.-conclusion-and-future-directions}

### 11.1 Summary of Contributions {#11.1-summary-of-contributions}

This paper has presented Score Vision, a novel decentralised protocol that addresses fundamental challenges in the deployment of production-ready computer vision systems. Our contributions span architectural innovation, economic mechanism design, security frameworks, and performance evaluation methodologies.

The two-tier Element-Agent architecture provides a principled approach to decomposing computer vision capabilities into atomic, verifiable units while enabling their composition into production-grade systems. This separation enables independent development and optimisation of specialised capabilities while maintaining system-level performance guarantees, addressing the composability challenges that plague monolithic model deployment.

The difficulty-weighted emission system with baseline gates and real-time constraints creates economic incentives that align with practical deployment requirements rather than pure accuracy optimisation. The dual-phase economic model provides a sustainable path from decentralised development to commercial deployment through USD-backed redemption pools, preserving competitive dynamics while enabling token value realisation.

Comprehensive anti-gaming mechanisms—including per-validator cryptographic salting, pseudo-ground truth generation with human audit validation, and TEE integration for privacy-preserving execution—demonstrate how decentralised evaluation networks can maintain integrity in adversarial environments. These mechanisms address both traditional gaming strategies and novel attack vectors specific to AI evaluation systems.

The Real-Time Factor (RTF) framework provides a hardware-agnostic method for ensuring that accuracy improvements translate to practical deployment viability. By normalising latency evaluation against declared service rates, the protocol addresses a critical gap in traditional machine learning benchmarks where solutions optimise for leaderboard performance while remaining impractical for production use.

### 11.2 Implications for Computer Vision and Potential Methodological Contributions {#11.2-implications-for-computer-vision-and-potential-methodological-contributions}

Score Vision demonstrates several important principles for production computer vision deployment that address long-standing challenges in making cameras truly intelligent at scale.

By designing atomic Elements with strict interface contracts, the protocol enables modular development that scales with system complexity. This composability-first approach contrasts sharply with monolithic model development where improvements in one component require retraining or redeploying entire systems. The Element architecture provides a path toward more maintainable and upgradeable vision systems where specialised capabilities can evolve independently while maintaining system-level guarantees.

The integration of real-time constraints directly into economic incentives ensures that research advances translate to practical improvements rather than pure benchmark optimisation. This performance-reality alignment addresses the persistent gap between academic competitions—where solutions often require impractical computational resources—and production requirements where latency, cost, and reliability constraints dominate. By making real-time viability a prerequisite for rewards, the protocol fundamentally redirects competitive energy toward practically deployable solutions.

The dual-lane architecture demonstrates how competitive development can coexist with commercial privacy requirements. The Public Track maintains open competition and transparent evaluation while the Trusted Track enables privacy-preserving execution for commercial deployments. This design enables a smooth transition from open research to proprietary deployment without compromising the competitive dynamics that drive innovation, addressing a major friction point in AI commercialisation.

Finally, the protocol's continuous evaluation and reward structure enables perpetual model improvement through ongoing competition. This contrasts fundamentally with traditional deployment models where systems are trained once, deployed, and then stagnate until manual intervention triggers retraining. Score Vision's design ensures that deployed vision systems benefit from continuous capability advancement driven by competitive pressure, moving toward the goal of cameras that become progressively more intelligent over time.

While these principles—atomic capability design, real-time performance integration, privacy-preserving competition, and continuous improvement—could potentially inform decentralised development approaches in other AI domains, Score's focus remains specifically on computer vision. Our mission is making every camera intelligent, not building general AI infrastructure. The methodological contributions we describe may prove valuable to others exploring decentralised AI networks, but they serve Score's singular vision of transforming how vision systems are built and deployed.

### 11.3 Limitations and Areas for Improvement {#11.3-limitations-and-areas-for-improvement}

Several limitations of the current design present opportunities for future research and protocol evolution.

The initial focus on sports video analysis, while providing a concrete validation domain with clear performance metrics and abundant training data, limits immediate applicability to other computer vision domains. Industrial monitoring, medical imaging, and autonomous systems present distinct challenges including different accuracy-latency trade-offs, domain-specific evaluation metrics, and varied privacy requirements. Expanding to these domains will require careful co-design of Elements, evaluation methodologies, and economic incentives appropriate to each vertical.

The current centralised governance model, while enabling rapid iteration and parameter tuning essential during early development, represents a potential single point of failure and may limit long-term community participation in protocol evolution. The path toward decentralised governance must balance community empowerment with protection against governance capture, parameter manipulation, and short-term optimisation that sacrifices long-term protocol health.

Reliance on pseudo-ground truth generation introduces potential systematic biases favoring certain architectural approaches, training methodologies, or data distributions. While multi-model ensembles, human audit, and discovery credits provide partial mitigation, the fundamental challenge of automated evaluation quality remains. More robust ground truth generation incorporating diverse evaluation paradigms and stronger geometric consistency constraints would strengthen the evaluation framework.

The Phase 2 commercial integration mechanisms—including redemption pool management, prudence factor calibration, and price bound determination—while theoretically sound, require real-world validation with actual commercial revenue. The transition from pure emissions to revenue-backed redemption must navigate volatility in commercial demand, potential gaming of redemption timing, and the challenge of maintaining fair pricing across Elements with vastly different commercial traction.

### 11.4 Future Research Directions {#11.4-future-research-directions}

Several promising research directions emerge from this work spanning domain expansion, evaluation methodologies, governance models, and technical enhancements.

Multi-domain expansion represents the most immediate research frontier. Extending the protocol to industrial computer vision applications—quality control, safety monitoring, process optimisation—would demonstrate Element-Agent architecture generalisability beyond entertainment. Medical imaging adaptation requires addressing additional privacy, regulatory, and safety constraints while maintaining competitive dynamics. Integration with robotics and autonomous vehicle perception would test the protocol's ability to handle safety-critical applications with stringent reliability requirements and real-time guarantees even more demanding than live sports.

Advanced evaluation methodologies could strengthen the protocol's ability to drive meaningful capability improvement. Incorporating adversarial robustness into multi-pillar metrics would incentivise development of vision systems resilient to distribution shift and adversarial perturbation. Integrating uncertainty quantification into Element contracts would enable more reliable system composition, allowing downstream Agents to make risk-aware decisions based on upstream Element confidence estimates.

Governance evolution must address the transition from centralised parameter control to community-driven mechanisms. Research directions include designing parameter adjustment proposals resistant to manipulation, exploring cross-subnet coordination with other Bittensor subnets to enhance computer vision capabilities, and adapting the protocol to meet evolving regulatory requirements for vision systems while preserving competitive dynamics.

Technical enhancements span multiple infrastructure layers. Advanced TEE integration expanding support for emerging trusted execution technologies and novel privacy-preserving computation methods would strengthen the Trusted Track. Federated learning integration could enable collaborative model improvement while maintaining competitive evaluation. More sophisticated scheduling and resource management algorithms for Vision Agents would maximise performance under resource constraints, particularly as Element complexity and Agent integration sophistication increase.

### 11.5 Vision and Impact {#11.5-vision-and-impact}

Score Vision's mission is making every camera intelligent. This vision drives our focus on creating decentralised infrastructure for production-ready computer vision that combines the innovation benefits of open competition with the practical requirements of real-world deployment. The protocol demonstrates how economic incentives, continuous evaluation, and composable architecture can transform vision system development from a one-time training process into an ongoing competitive improvement cycle.

The long-term vision encompasses ubiquitous intelligent camera systems—from sports analytics and broadcast production to industrial monitoring, retail analytics, security applications, and urban infrastructure. By creating economic mechanisms that reward practical utility over benchmark performance, standardised interfaces that enable reliable composition, and continuous competition that drives perpetual improvement, we aim to make sophisticated computer vision accessible and continuously improving across every application domain. This vision is specifically about cameras and vision systems, not general AI infrastructure, though the methodological approaches we pioneer may inform others working on adjacent challenges.

### 11.6 Final Remarks {#11.6-final-remarks}

The Score Vision protocol embodies a simple but powerful principle: align economic incentives with practical requirements, maintain transparency and verifiability, and let competitive dynamics drive continuous improvement. By publishing clear rules, running standardised evaluations, rewarding genuine improvements, and burning underperformance, the protocol creates a sustainable mechanism for advancing computer vision capabilities under real-world constraints.

The success of this approach will ultimately be measured not by the sophistication of its mechanisms, but by the practical utility of the vision systems it produces and their impact on making cameras truly intelligent. As the protocol evolves and expands across application domains—from sports and entertainment to industry, security, and urban infrastructure—we aim to demonstrate how decentralised competition can deliver production-ready computer vision at scale.

Making every camera intelligent requires systems that combine the best aspects of open research, competitive development, and practical deployment. Score Vision provides an architecture and economic model specifically designed for this vision, creating infrastructure where computer vision capabilities become continuously improving, composable building blocks that serve real-world applications. The goal is not to revolutionise AI broadly, but to transform computer vision specifically—making intelligent perception accessible, reliable, and continuously advancing wherever cameras operate.

---

## Appendix A — JSON Schemas (abbrev) {#appendix-a-—-json-schemas-(abbrev)}

- `Detections[]`: `{ frame_idx:int, class:string, bbox:[float;4], score:float, track_id:string, team_id:int?, role:string? }`
- `Keypoints[]`: `{ frame_idx:int, points:[{id:string,x:float,y:float}], homography:[[float;3];3]? }`
- `XY[]`: `{ frame_idx:int, track_id:string, xy:[float;2], unit:"meters"|"pixels" }`

Full machine‑readable JSON Schemas ship with the SDK.

---

## Appendix B — Parameter Tables (defaults) {#appendix-b-—-parameter-tables-(defaults)}

**Elements**

- PlayerDetect_v1: p95 ≤ 200 ms @ 5 FPS; β \= 1.0; baseline θ_player \= published per window; service rate 25 FPS
- BallDetect_v1: p95 ≤ 200 ms @ 5 FPS; β \= 1.4; θ_ball \= … ; service rate 25 FPS
- PitchCalib_v1: p95 ≤ 250 ms @ 5 FPS; β \= 0.9; θ_calib \= … ; service rate 5 FPS

**Agent (public integration test)**

- Gate: RTF ≤ 1.0 (composed), jitter ≤ 40 ms, memory ≤ 6000 MB, continuity ≥ 0.95 → \+5% Element earnings (bonus)

**Economics**

- γ_trusted: published per window
- Discovery credit `G_i`: 1.00–1.03
- Concentration guard: τ \= 30% (example), λ \= 1 (burn excess)

---

## Appendix C — Math {#appendix-c-—-math}

**EWMA:** `α = 1 − 2^(−1/h)`; with `h=3`, `α≈0.2063`. **RTF:** `RTF_e = (t_p95_ms / 1000) * (r_e / 5)`; pass if `≤ 1.0`.

---

## Appendix D — Manifest Snippet (illustrative) {#appendix-d-—-manifest-snippet-(illustrative)}

```json
{
  "window_id": "2025-10-27",
  "elements": [
    {
      "id": "PlayerDetect_v1@1.0",
      "clips": ["sha256:..."],
      "weights": [1.0],
      "preproc": { "fps": 5, "resize_long": 1280, "norm": "rgb-01" },
      "metrics": {
        "pillars": {
          "iou": 0.35,
          "count": 0.2,
          "palette": 0.15,
          "smoothness": 0.15,
          "role": 0.15
        }
      },
      "latency_p95_ms": 200,
      "service_rate_fps": 25,
      "salt": { "offsets": [0, 1, 2, 3, 4], "strides": [5, 6] },
      "pgt_recipe_hash": "sha256:...",
      "baseline_theta": 0.78,
      "delta_floor": 0.01,
      "beta": 1.0
    }
  ],
  "tee": { "trusted_share_gamma": 0.2 },
  "version": "1.3",
  "expiry_block": 123456
}
```

---

## Appendix E — Glossary {#appendix-e-—-glossary}

- **Element:** Atomic, scored capability unit (e.g., PlayerDetect, BallDetect, PitchCalib) with strict I/O contracts and real-time performance requirements.
- **Agent:** Orchestrated pipeline integrating multiple Elements into production-grade systems with end-to-end SLOs.
- **Manifest:** Cryptographically signed per-window rulebook specifying Elements, metrics, baselines, latency gates, and PGT recipes.
- **PGT (Pseudo-Ground Truth):** Validator-local reference annotations generated by VLM ensembles; miners never observe PGT during evaluation.
- **RTF (Real-Time Factor):** Hardware-agnostic latency metric; RTF ≤ 1.0 means viable at the declared service rate. Formula: `(t_p95_ms / 1000) × (r_e / 5)`.
- **Canary Gold:** Small human-labeled dataset anchoring PGT quality and enabling disagreement detection.
- **β (Beta / Difficulty Weight):** Emission multiplier tilting rewards toward harder, scarcer capabilities.
- **θ (Theta / Baseline):** Minimum composite score threshold before earning rewards; performance below θ receives zero emissions.
- **δ (Delta Floor):** Minimum improvement margin required above baseline to earn rewards.
- **EWMA (Exponentially Weighted Moving Average):** Temporal smoothing of scores across evaluation windows; default half-life h=3.
- **VLM (Vision-Language Model):** Large multimodal models (e.g., Moondream 3 Preview, Qwen2.5-VL-72B, InternVL3-78B) used for PGT generation and baseline establishment.
- **TEE (Trusted Execution Environment):** Hardware-isolated secure enclaves (Chutes/SN64, Targon/SN4, AWS Nitro, Intel SGX, AMD SEV-SNP, NVIDIA CC) enabling privacy-preserving execution.
- **Public Track:** Transparent lane where miners compete openly; all evaluation data contributes to protocol improvement.
- **Trusted Track:** Privacy-preserving lane using TEEs; no raw frames or model weights leave secure enclaves.
- **Discovery Credits:** Reward bonuses (1.00–1.03×) for miners who identify PGT errors validated through human audit.
- **Goldilocks Redemption:** Phase 2 economic mechanism converting commercial USD revenue into Element-specific token redemption pools.
- **VRF/PRF (Verifiable/Pseudo-Random Function):** Cryptographic primitives enabling unpredictable per-validator challenge salting.
- **Score Cloud:** Commercial platform providing unified API access to network capabilities with dual execution modes (Public/Trusted).
- **Forward Reasoning:** Future protocol phase enabling Elements that predict events based on compositional analysis of upstream outputs.
- **Multi-Pillar Metrics:** Evaluation framework combining multiple orthogonal dimensions (IoU, count accuracy, palette symmetry, smoothness, role consistency) to resist gaming.

---

## Appendix F — References {#appendix-f-—-references}

### Computer Vision Models and Datasets {#computer-vision-models-and-datasets}

- [**1**](https://bittensor.com/dtao-whitepaper) **YOLO (You Only Look Once)** — Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection" (2016). [https://arxiv.org/abs/1506.02640](https://arxiv.org/abs/1506.02640)
- **\[2\] DETR** — Carion et al., "End-to-End Object Detection with Transformers" (2020). [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
- **\[3\] ByteTrack** — Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (2021). [https://arxiv.org/abs/2110.06864](https://arxiv.org/abs/2110.06864)
- **\[4\] DeepSORT** — Wojke et al., "Simple Online and Realtime Tracking with a Deep Association Metric" (2017). [https://arxiv.org/abs/1703.07402](https://arxiv.org/abs/1703.07402)
- **\[5\] COCO Dataset** — Lin et al., "Microsoft COCO: Common Objects in Context" (2014). [https://arxiv.org/abs/1405.0312](https://arxiv.org/abs/1405.0312)
- **\[6\] ImageNet** — Russakovsky et al., "ImageNet Large Scale Visual Recognition Challenge" (2015). [https://arxiv.org/abs/1409.0575](https://arxiv.org/abs/1409.0575)
- **\[7\] MOTChallenge** — Dendorfer et al., "MOTChallenge: A Benchmark for Single-Camera Multiple Target Tracking" (2020). [https://arxiv.org/abs/2010.07548](https://arxiv.org/abs/2010.07548)

### Computer Vision Frameworks {#computer-vision-frameworks}

- **\[8\] Detectron2** — Facebook AI Research. [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
- **\[9\] OpenMMLab** — Comprehensive computer vision framework. [https://openmmlab.com/](https://openmmlab.com/)
- **\[10\] PaddleDetection** — Object detection toolkit. [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

### Industry Platforms {#industry-platforms}

- **\[11\] NVIDIA Metropolis** — Intelligent video analytics platform. [https://www.nvidia.com/en-us/autonomous-machines/intelligent-video-analytics-platform/](https://www.nvidia.com/en-us/autonomous-machines/intelligent-video-analytics-platform/)
- **\[12\] DeepStream SDK** — Multi-camera tracking and pipeline. [https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html)
- **\[13\] NVIDIA NIM** — Inference microservices. [https://docs.nvidia.com/nim/index.html](https://docs.nvidia.com/nim/index.html)
- **\[14\] NVIDIA TAO Toolkit** — Fine-tuning for vision foundation models. [https://docs.nvidia.com/tao/tao-toolkit/index.html](https://docs.nvidia.com/tao/tao-toolkit/index.html)

### Trusted Execution Environments {#trusted-execution-environments}

- **\[15\] AWS Nitro Enclaves** — Cryptographic attestation documentation. [https://docs.aws.amazon.com/enclaves/latest/user/set-up-attestation.html](https://docs.aws.amazon.com/enclaves/latest/user/set-up-attestation.html)
- **\[16\] Intel SGX DCAP** — Data Center Attestation Primitives orientation. [https://www.intel.com/content/dam/develop/public/us/en/documents/intel-sgx-dcap-ecdsa-orientation.pdf](https://www.intel.com/content/dam/develop/public/us/en/documents/intel-sgx-dcap-ecdsa-orientation.pdf)
- **\[17\] AMD SEV-SNP** — Strengthening VM isolation with integrity protection. [https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/SEV-SNP-strengthening-vm-isolation-with-integrity-protection-and-more.pdf](https://www.amd.com/content/dam/amd/en/documents/epyc-business-docs/white-papers/SEV-SNP-strengthening-vm-isolation-with-integrity-protection-and-more.pdf)
- **\[18\] NVIDIA Confidential Computing** — H100 GPUs for secure AI. [https://developer.nvidia.com/blog/confidential-computing-on-h100-gpus-for-secure-and-trustworthy-ai/](https://developer.nvidia.com/blog/confidential-computing-on-h100-gpus-for-secure-and-trustworthy-ai/)

### Decentralised AI Networks {#decentralised-ai-networks}

- **\[19\] Fetch.ai** — Agent and service marketplace platform. [https://fetch.ai/docs](https://fetch.ai/docs)
- **\[20\] SingularityNET** — Decentralised AI marketplace. [https://singularitynet.io/](https://singularitynet.io/)
- **\[21\] Ocean Protocol** — Data marketplace protocol. [https://docs.oceanprotocol.com/](https://docs.oceanprotocol.com/)

### Bittensor Subnets {#bittensor-subnets}

- **\[22\] Chutes (SN64)** — TEE subnet for Score Vision Trusted Track. [https://www.tao.app/subnets/64](https://www.tao.app/subnets/64)
- **\[23\] Targon (SN4)** — TEE subnet for privacy-preserving execution. [https://www.tao.app/subnets/4](https://www.tao.app/subnets/4)

### Vision-Language Models {#vision-language-models}

- **\[24\] Qwen2.5-VL-72B-Instruct** — State-of-the-art open-source VLM. [https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
- **\[25\] InternVL3-78B** — Frontier vision-language model. [https://huggingface.co/OpenGVLab/InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B)
- **\[26\] Moondream 3 Preview** — Mixture-of-experts VLM (9B total / 2B active params, 32k context). [https://huggingface.co/moondream/moondream3-preview](https://huggingface.co/moondream/moondream3-preview) and [https://moondream.ai/blog/moondream-3-preview](https://moondream.ai/blog/moondream-3-preview)

### Annotation and Quality Assurance {#annotation-and-quality-assurance}

- **\[27\] Label Studio** — Open-source data labeling platform. [https://labelstud.io/](https://labelstud.io/)
- **\[28\] CVAT** — Computer Vision Annotation Tool. [https://docs.cvat.ai/docs/](https://docs.cvat.ai/docs/)

### Protocol and Network Infrastructure {#protocol-and-network-infrastructure}

- **Dynamic TAO (dTAO) Whitepaper** — Canonical emission, halving, and validator mechanics; Score Vision inherits base monetary policy from dTAO. ([bittensor.com](https://bittensor.com/dtao-whitepaper))

---
