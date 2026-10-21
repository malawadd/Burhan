# Competition and cross market screening evidence

Research date: 20 October 2026. Scope: global. Constraints supplied by the user: no partner, hardware, or private data access. A laptop and internet connection are planning assumptions; team size, skills, AWS account, and spend budget remain unknown.

## Competition verification

The organizer's [Devpost overview](https://opencv26.devpost.com/) requires substantive OpenCV 5 analysis and meaningful AWS execution. Its narrative specifies 26 October at 11:59 p.m. Pacific, while the platform header displays 11:45 p.m. PDT. Plan to submit before the earlier displayed cutoff. The [rules](https://opencv26.devpost.com/rules) confirm optional special awards: a project need not use COOL or an agent to compete for an overall award. The supplied brief is the source for detailed deliverable and judging requirements in the final study.

The [public gallery](https://opencv26.devpost.com/project-gallery) explicitly says the managers have not published it. There is no evidence for claiming an idea is unique among current entrants. Market competition can be assessed; the submission field cannot yet be assessed.

OpenCV's [5.0.0 release page](https://opencv.org/release/opencv-5-0-0/) confirms the required major version exists. Pin a working build and verify `cv2.__version__` during implementation. An LLM interpreting an image alone does not establish substantive OpenCV work. The [OpenCV 5 DNN engine documentation](https://docs.opencv.org/doc/doxygen/html/de/db7/api_dnn_engine_selection.html) recommends ONNX and documents engine differences; model compatibility must be tested rather than assumed.

## Additional sector screened commercial kitchen food waste

There is strong customer-side evidence of an operational need. [Ingka Group, 12 May 2026](https://www.ingka.com/newsroom/one-million-bags-of-leftovers-and-how-were-putting-them-to-good-use/) reports production food waste per cover down 60% since FY17 and Waste Watcher installed in 402 stores by FY25. This is a specific adopter outcome involving a long-running operational program and Winnow, not a controlled estimate of an image classifier's effect. It establishes adoption, not a greenfield opportunity.

Competition is substantial. [Winnow](https://www.winnowsolutions.com/why-winnow) combines capture, analytics, and operational guidance. [Leanpath's product suite](https://www.leanpath.com/products/food-waste-tracking/) includes scale-based and mobile camera-based capture; consequently, a camera-only proposition is not a demonstrated gap. [Orbisk's published plans](https://orbisk.com/product/solutions/) include ingredient recognition, reporting, and a culinary advisor; its page explicitly says pricing is scoped by sales, with hardware included in subscriptions. Do not infer inexpensive entry pricing from absent public figures.

Likely users are kitchen teams; a food and beverage director, operations director, or contract catering account director would plausibly own the budget. Ingka, Metz Culinary Management, Marriott, and Accor appear in operator or vendor material as category adopters; these are incumbent-served organizations, not validated leads. Small independent kitchens may have different economics and willingness to use another capture process. No evidence in this scan establishes that their unmet need is solved by a new generic food classifier.

Disposition: do not prioritize under current constraints. There is no kitchen access or meal/waste/weight ground truth. A food photo benchmark does not validate mass measurement, avoided purchases, or staff adherence. No quantified savings should be promised from segmentation alone. Revisit only with a partner who supplies usable before-and-after waste records and identifies a problem their current process leaves unresolved.

## Cloud architecture and cost evidence

A proposed minimum architecture is browser upload to private S3, a container on EC2 performing the OpenCV workload and workflow routing, and a review interface with evidence and decision traces. [AWS documents time-limited S3 presigned uploads and downloads](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html). This architecture is a design proposal, not an already-deployed service. No cloud resources were created during research.

[AWS's T3 product page](https://aws.amazon.com/ec2/instance-types/t3/) displays t3.medium at $0.0418 per hour in its example table; pricing depends on region and operating system, with potential burst credit charges. An illustrative compute-only calculation is 168 hours times $0.0418 = $7.02. [AWS VPC pricing](https://aws.amazon.com/vpc/pricing/) lists $0.005 per in-use public IPv4 hour, or $0.84 for 168 hours. Storage, requests, EBS, logs, data transfer, taxes, optional models, and CPU credits are additional. A $20-40 prototype allowance is an assumption for planning and is neither a measured bill nor a guaranteed cap. Confirm region pricing and workload memory/CPU needs before deployment. Do not assume a grant or free-tier eligibility.

## Evidence standard

Operator reports establish actual adoption and observed operational problems; vendor pages establish advertised capabilities, not independently measured accuracy. Public research data establish a route to reproducible evaluation, not customer validation. Named organizations are prospective interview targets unless a source explicitly identifies them as existing category adopters. Commercial demand for any proposed niche, price, integration, and claimed benefit requires future interviews and pilots.
