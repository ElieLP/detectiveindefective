# Topic D: AI Copilot for Industrial Quality Teams

Sponsor: Safran Aircraft Engines Suzhou (use 'Safran' for communication)

---

## 🎯 Challenge Overview

The Problem: Every factory produces Non-Conformity Reports (NCRs). Hidden inside these reports are:

- Recurring machine failures
- Weak suppliers
- Early warning signals before bigger incidents happen

Your Mission: Explore how AI can support NCR (Non-Conformity Report) analysis to:

- Reduce NCR processing time
- Identify recurring defect patterns (machines, suppliers, process steps)
- Support root cause analysis
- Suggest corrective actions based on similar cases
- Provide quick summaries
- Generate simple dashboards showing trends and weak signals

---

## 👥 Target Users & Context

- Environment: Internal use in an industrial environment
- Context: The product is used internally, in an industrial environment

---

## 📦 Provided Resources

### Dataset A: 100 NCRs

- Purpose: Provide enough diversity without overloading teams
- Scope: Limited to production line only, no assembly line for now
- Format: Excel or CSV file
- Fields include: NCR_ID, Description, Machine, Supplier, Operator, Category (HOF, process, maintenance, training, documentation issue, etc.), Root Cause, Corrective Action

### Anonymization

Sensitive information will be anonymized with neutral tokens:

- Machines → MACHINE_01, MACHINE_02, etc.
- Operators → OP_01, OP_02, etc.
- Suppliers → SUPPLIER_A, SUPPLIER_B, etc.
- Clients → CLIENT_X, CLIENT_Y, etc.
- Part numbers → PN_XXXX
- Serial numbers → SN_XX1

⚠️ Important: The technical content, defects, causes and actions will NOT be modified. It's important to train the model with raw data.

### Dataset B: 10-20 High-Quality "Gold Sample" NCRs

- Purpose: Help benchmark the suggestions

---

## ⚖️ Usage Terms & Compliance

- Data Usage: As data are anonymized, candidates can use the dataset on any tools and on 3rd party cloud
- Data Retention: However, data should NOT be kept or reused after the Hackathon
- Intellectual Property: The intellectual property of the POC, solutions or all content created for this challenge remains with Safran

---

## 🚀 Challenge Phases

### Phase 1: Core Challenge (Minimum Viable Product)

Objective: Develop a prototype demonstrating core capabilities.

Expected Capabilities:

- Analyze: Extract key elements from NCR text (defect, operation, machine, supplier)
- Classify: Group NCRs by similarity or category and identify recurring patterns
- Recommend: Suggest likely root causes and propose corrective actions based on past cases

---

### Phase 2: Improvement & Usability

Objective: Make your solution more useful and valuable for quality teams.

Consider:

- How can quality teams visualize and understand the insights?
- What questions do they need answered quickly?
- How does this fit into their daily workflow?
- What value does this provide beyond just analysis?

---

### Phase 3: Advanced / Innovative Features

Objective: Push the boundaries and demonstrate real-world potential.

Consider:

- How do quality teams and shop-floor workers actually use this?
- How could this scale across multiple production lines or sites?
- What would make this solution indispensable to quality teams?
- How do you demonstrate the value - time savings, cost reduction, defect prevention?
- What's the path from prototype to full deployment?

---

## 📋 Expected Deliverables

At the end of the hackathon, you should submit a complete solution that demonstrates:

1. Working Prototype:
   - Core AI capabilities (analyze, classify, recommend)
   - Source code or notebooks
2. A Product That Solves the Problem:
   - How do quality teams use this solution?
   - What makes it valuable and deployable?
   - How does it demonstrate real-world impact?
3. Presentation:
   - 5-7 minute demo showcasing your solution
   - Clear communication of the value and impact

Think creatively: Build a complete product, not just an analysis tool. Consider how users interact with it, what value it provides, and how it could be deployed in real manufacturing environments.

Important Notes:

- The scope should be limited but impactful
- The output should be suitable for a 5-7 minute demo
- Data must be anonymized
- Only a small dataset should be shared

Remember: This POC is not meant to be a final product. The goal is to test the value and feasibility of an AI solution that supports quality and manufacturing teams, especially with the upcoming production ramp-up.

---

## 🏆 Evaluation Criteria

Prototypes will be evaluated based on:

- Usefulness
- Accuracy
- Clarity
- Potential for future industrialization

---

## 💡 Tips for Success

### Focus on Practical Value:

- The solution should be useful for quality and manufacturing teams
- Data is key: Use the raw technical content (defects, causes, actions) effectively - it's not anonymized for a reason
- Keep it simple: This is a POC, not a final product - focus on demonstrating value and feasibility

### Think About the Complete Product:

- Who uses this? Quality engineers and shop-floor workers - how do they interact with your solution?
- What's the value? Time savings, cost reduction, defect prevention - how do you demonstrate this?
- How does it work in practice? Consider real-world deployment - integration, scalability, user adoption
- How do you communicate impact? Defect prevention and quality improvement matter - how do you tell that story?

### Key Questions to Consider:

- How do quality teams visualize and understand the insights?
- What makes this solution indispensable to quality teams?
- How could this scale across multiple production lines or sites?
- What's the path from prototype to full deployment?
- How do you demonstrate ROI and impact to management?

---

## ❓ Questions?

If you have questions about this topic, please ask them in the WeChat group for Topic D.