# LawConnect Engineering Take-Home Task (External)

## Scenario

An individual **Michelle Anne Ritchie** was dismissed from Northern Rivers Allied Health. She wants to review her available records to understand whether they support a potential employment-law claim before deciding whether to seek legal advice.

The task can be decomposed into 2 main stages:
### Stage 1: organised evidence package

To first get on the same page as the prospective client's understanding of their situation, we want to search through their data sources to see what relevant material they have to the case at hand. For the particular scenario above, all related evidence should be organised into logical folders given the type of legal matter at hand. 

One naive example of 'logical' folders would be "gmail evidence" and "google drive evidence".

The output can be a folder on disk, a zip file delivered over http, etc. - anything that is applicable/appropriate depending on how you build the service.

You may choose any medium to deliver this service - it need not have prod-level polish. It could be a cli, a basic web app, etc.
### Stage 2: anonymous summary

Once the user is satisfied that the organised evidence package contains all their related evidence, they should then be given the option of having an anonymous summary created for them that could feasibly be presented on a legal marketplace where law firms/lawyers can decide to take on a case or not based on such a summary.

The anonymised summary should retain details that are important (e.g. in employment law, the high income threshold is often referenced and is currently $190,100 as of 1 Jul 2026). However, it should *not* include trivially identifying information such as names, employers addresses, or other *directly* identifying details.

## Evidence source

A zip file of a mix of related and unrelated evidence has been provided for you to work against. It contains various kinds of documents, and also emails with attachments. 

Your system should ingest these from either local disk directly, *or* a combination of a cloud drive+email service (e.g. Gmail + Gdrive).
## What to build

Build a working system that:

1. collects evidence from your chosen source;
2. organises the relevant evidence into structured folders
3. presents the option to the user to prepare an anonymous summary and does so with their approval

You may build a command-line tool, web application, API, agent tool, any combination of these, or another interface that suits your design.

## Deliverables

Please provide:

1. working code in a git repository with clear setup and run instructions;
2. a README explaining:
   - the architecture;
   - the main decisions and trade-offs;
   - what anonymisation means in your system;
   - what are the exit conditions for both stages;
   - known limitations; 
   - any assumptions
   - and a brief overview of the AI tools/setup you used for the task (free reign to use as much or as little here as you are comfortable with)
3. a brief video recording demoing the final result

Please make the submission runnable with one documented command after dependencies have been installed.

## Timebox

We would encourage you to spend approximately 8 hours on the task. 

## Out of scope

The scenario brief mentions that the intention of the summary + evidence is to hand over to a law firm. Receipt of documents by the law firm/lawyer is out of scope of this task.

## Questions

You may choose to ask clarifying questions before or during the task (contact **Justin Ting** via Whatsapp at **0494 198 955**). Alternatively you are welcome to make assumptions regarding anything in the task -  please record these as part of the submission.