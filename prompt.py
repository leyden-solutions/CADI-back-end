
sections_prompt = """
    You are an expert at carefully analyzing government classification guidelines and extracting the RELEVANT sections from the document. RELEVANT sections are the sections that contain tables corresponding to classification guidelines. D Given a full guideline document, you will return a list of strings representing the RELEVANT sections in the document. NON-RELEVANT-SECTIONS include appendices, introductions, paragraphs, etc. Only focus on what you are confident is a classification guideline rule. only include the numbers, not the title of the section.

    Here is the JSON schema:

    {
        "sections": [
            "section_1",
            "section_2",
            "section_3"
        ]
    }

    Example:
    {
        "sections": [
            "3.1.1",
            "3.1.2",
            "3.1.3",
            ...
        ]
    }

    Ensure you identify ALL sections in the document, do not skip a single one. Be very thorough and careful.
"""








guidelines_prompt = """

You are an expert at formatting government classification guideline TABLES  into a structured JSON format. You will be given a full document and you will need to format a section into a structured JSON format. You must follow the schema that I want to format the class guides into EXCATLY, make sure to convert the duration value into an integer of years.
Here is the JSON schema:

{
"#.#.#": {
    "name": "name"

        items: [

            {

                content: "Info from item column",

                level: "info from level column",

                duration: "info from declass on column",

                reason: "info from reason column",

                remarks "info from the remarks column"

            },

            {

                content: "Info from item column",

                level: "info from level column",

                duration: "info from declass on column",

                reason: "info from reason column",

                remarks "info from the remarks column"

            }
            ...
        ]

    }

}

Other information to note:
- The document provides information like "Current date+25 years" or "25X1+50 years". Sa, you should  always extract the integer value following the '+' sign (e.g., "25" from "Current date+25 years" and "50" from "25X1+50 years"),
- For values that are empty have them as null instead of None because this is json not a python dictionary
- Ensure you have the duration as an INTEGER ALL THE TIME
- Focus  on providing the most accurate and correctly formatted output possible at all times.
- Ensure you finish all rules in their entirety, your response should extensive, take as much time as you need to analyze the document extensively and provide a complete response.
- Do not provide any external text or formatting strings, only the JSON object.


example:
    "3.3.1": {
        "name": "(U) FOREIGN INTELLIGENCE SURVEILLANCE ACT (FISA)",
        "items": [
            {
                "content": "(U) The fact that NCTC uses FISA-acquired and FISA-derived information in its analysis and in its intelligence products.",
                "level": "U",
                "duration": null,
                "reason": "(U) UNCLASSIFIED when used generically.",
                "remarks": null
            },
            {
                "content": "(U) Information obtained by, or derived from, an investigative technique requiring a FISA Court order or other FISA authorized collection (\"FISA Information\").",
                "level": "S",
                "derivative": "ODNI FISA S-14",
                "dissemination_control": "NOFORN//FISA",
                "reason": "1.4(c)",
                "duration": "Current date+ 25 years",
                "oca": "FBI",
                "remarks": "(U) May be classified higher based on individual collection agency or assisting agency classification guidance."
            },
            {
                "content": "(U) Use of FISA-acquired and FISA-derived information when connected to a particular target or analytic judgment in intelligence products or analysis.",
                "level": "S",
                "derivative": "ODNI FISA S-14",
                "dissemination_control": "NOFORN//FISA",
                "reason": "1.4(c)",
                "duration": "Current date + 25 years",
                "oca": null,
                "remarks": "(U) May be classified higher based on individual collection agency classification guidance."
            },
            {
                "content": "(S//NF) The fact of NCTC's access to and use of raw, unminimized FISA-acquired information.",
                "level": "S",
                "derivative": "ODNI FISAS-14",
                "dissemination_control": "NOFORN",
                "reason": "1.4(c)",
                "duration": "Current",
                "oca": null,
                "remarks": null
            },
            {
                "content": "(S//NF) The process and procedures of making raw, unminimized FISA-acquired information available to NCTC personnel.",
                "level": "S",
                "derivative": "ODNI FISA S-14",
                "dissemination_control": "NOFORN",
                "reason": "1.4(c)",
                "duration": "Current",
                "oca": null,
                "remarks": null
            },
            {
                "content": "(S//NF) NCTC's Standard Minimization Procedures.",
                "level": "S",
                "derivative": "ODNI FISA S-14",
                "dissemination_control": "NOFORN",
                "reason": "1.4(c)",
                "duration": "Current",
                "oca": "AG",
                "remarks": null
            },
            {
                "content": "(S//NF) The policies, process, and procedures for implementing NCTC's Standard Minimization Procedures.",
                "level": "S",
                "derivative": "ODNI FISA S-14",
                "dissemination_control": "NOFORN",
                "reason": "1.4(c)",
                "duration": "Current",
                "oca": null,
                "remarks": null
            },
            {
                "content": "(S//NF) The fact that NCTC employees are required to sign a Letter of Consent to View and Work with Objectionable Material in order to access unminimized FISA acquired information.",
                "level": "S",
                "derivative": "ODNI FISA S-14",
                "dissemination_control": "NOFORN",
                "reason": "1.4(c)",
                "duration": "Current date + 25 years",
                "oca": null,
                "remarks": null
            },
            ...

        ]
    },

You will only process tables a few sections at a time. The sections to process next are:
{sections}
Only process results from these sections, and no other sections. For example, if the sections are 3.2.1, 3.2.2, 3.2.3, the keys/sections should be 3.2.1, 3.2.2, 3.2.3.
"""

example_response = """
{
    "3.6.2": {
        "name": "(U) SECURITY (SEC)",
        "items": [
            {
                "content": "(U) Fact that an IC Agency or element has an insider threat detection program.",
                "level": "U",
                "duration": null,
                "reason": null,
                "remarks": "(U) Details may be higher."
            },
            {
                "content": "(U) Fact that NCIX, as part of the NITTF, conducts National Insider Threat Policy and Minimum Standards assessments and reviews for USG agencies.",
                "level": "U",
                "duration": null,
                "reason": null,
                "remarks": "(U) Classify details based on content and in accordance with applicable guidance."
            },
            {
                "content": "(U) The total number of DNI sponsored SCI control systems.",
                "level": "S",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) Examples: SI, TK, KDK, etc. (U) Contact P&S for further information and guidance."
            },
            {
                "content": "(U) General information regarding local disposal procedures for ODNI classified material, classified waste, and excess material.",
                "level": "U",
                "duration": null,
                "reason": null,
                "remarks": null
            },
            {
                "content": "(U) Specific disposal procedures for ODNI classified material, classified waste, and excess material outside of controlled areas (i.e., transportation route for disposal).",
                "level": "C",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": null
            },
            {
                "content": "(U) General information regarding the unauthorized disclosure of US classified information, which does not cite the classified information disclosed. Includes notifications of a spill or leak and activity reports.",
                "level": "U",
                "duration": null,
                "reason": null,
                "remarks": "(U) May be higher; consult appropriate Program Security Guide."
            },
            {
                "content": "(U) Information detailing damage assessment and supporting documentation for an unauthorized disclosure of US information that cites the compromised information deemed to be CONFIDENTIAL by the victim agency.",
                "level": "C",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) Contact ODNI Partner Engagement (ODNI/PE) for foreign disclosure guidance who will coordinate with NCIX regarding those aspects of the damage assessment which may be released to foreign partners."
            },
            {
                "content": "(U) Information detailing damage assessment and supporting documentation for an unauthorized disclosure of US information that cites the compromised information deemed to be SECRET by the victim agency.",
                "level": "S",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) Contact ODNI Partner Engagement (ODNI/PE) for foreign disclosure guidance who will coordinate with NCIX regarding those aspects of the damage assessment which may be released to foreign partners."
            },
            {
                "content": "(U) Information detailing damage assessment and supporting documentation for an unauthorized disclosure of US information that cites the compromised information deemed to be TOP SECRET by the victim agency.",
                "level": "TS",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) Contact ODNI Partner Engagement (ODNI/PE) for foreign disclosure guidance who will coordinate with NCIX regarding those aspects of the damage assessment which may be released to foreign partners."
            },
            {
                "content": "(U) Security-related documents of SCIF accreditations or technical security-related accreditations and associated documentation.",
                "level": "C",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) May be downgraded to FOUO if SCIF has been de-accredited and will not be used for another mission or project."
            },
            {
                "content": "(U) Information concerning the route, frequency, or mode of travel of courier runs.",
                "level": "C",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": null
            },
            {
                "content": "(U) Information detailing protective security and counterintelligence activities, equipment, techniques or tactics.",
                "level": "C",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": null
            },
            {
                "content": "(U) Information detailing non-attributable security methods.",
                "level": "C",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": null
            },
            {
                "content": "(U) Information pertaining to routine and covert personnel security and/or counterintelligence investigations where disclosure would impede or negate those investigations and/or risk leaving the ODNI vulnerable from a counterintelligence standpoint.",
                "level": "S",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": null
            },
            {
                "content": "(U) Information concerning potential or validated foreign intelligence service (FIS) threats to IC installations or personnel.",
                "level": "S",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) Additional markings may be required for compartmented information or programs. Check with appropriate Program Security Officer / Program Compartment Guide for compartmented information."
            },
            {
                "content": "(U) Information detailing physical security devices, techniques, policy, procedures, assessments, or mechanisms for protecting SECRET-level material where disclosure could provide the means to negate all or part of the protection intended.",
                "level": "S",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) Includes:\\n- Intrusion Detection Systems\\n- Security in Depth procedures\\n- Cryptographic components\\n- Alarm systems\\n- Technical Surveillance Counter Measures"
            },
            {
                "content": "(U) Information detailing physical security devices, techniques, policy, procedures, assessments, or mechanisms for protecting TOP SECRET-level material where disclosure could provide the means to negate all or part of the protection intended.",
                "level": "TS",
                "duration": 25,
                "reason": "1.4(c)",
                "remarks": "(U) Includes:\\n- Intrusion Detection Systems\\n- Security in Depth procedures\\n- Cryptographic components\\n- Alarm systems\\n- Technical Surveillance Counter Measures"
            }
        ]
    }
}
"""
