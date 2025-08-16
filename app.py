import os
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import PyPDF2
import docx2txt
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JobRequirement:
    """Represents a parsed job requirement"""
    category: str  # technical, soft_skill, experience, education, certification
    skill: str
    importance: str  # required, preferred, nice_to_have
    context: str

@dataclass
class CVSection:
    """Represents a section of the CV"""
    title: str
    content: str
    is_modifiable: bool  # Can this section be enhanced?
    keywords: List[str]

@dataclass
class EnhancementSuggestion:
    """Represents an enhancement suggestion from vector search"""
    code_snippet: str
    technologies: List[str]
    description: str
    relevance_score: float
    file_path: str
    repo_url: str

class DocumentProcessor:
    """Handles CV document processing (PDF, DOCX, TXT)"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def extract_text_from_cv(self, cv_path: str) -> str:
        """Extract text from CV file"""
        cv_path = Path(cv_path)
        
        if not cv_path.exists():
            raise FileNotFoundError(f"CV file not found: {cv_path}")
        
        file_extension = cv_path.suffix.lower()
        
        if file_extension == '.pdf':
            return self._extract_from_pdf(cv_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(cv_path)
        elif file_extension == '.txt':
            return self._extract_from_txt(cv_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _extract_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    def _extract_from_docx(self, docx_path: Path) -> str:
        """Extract text from DOCX"""
        try:
            return docx2txt.process(str(docx_path))
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
            raise
    
    def _extract_from_txt(self, txt_path: Path) -> str:
        """Extract text from TXT"""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT: {e}")
            raise

class JobDescriptionAnalyzer:
    """Analyzes job descriptions to extract requirements and keywords"""
    
    def __init__(self):
        self.technical_keywords = {
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring', 'hibernate',
            'sql', 'postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform',
            'git', 'jenkins', 'gitlab', 'github', 'ci/cd', 'devops',
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'pandas', 'numpy', 'matplotlib', 'jupyter', 'data science',
            'api', 'rest', 'graphql', 'microservices', 'agile', 'scrum',
            'linux', 'unix', 'bash', 'shell', 'automation', 'testing'
        }
        
        self.soft_skills = {
            'communication', 'leadership', 'teamwork', 'problem solving',
            'analytical', 'creative', 'adaptable', 'collaborative', 'initiative',
            'time management', 'project management', 'mentoring', 'presentation'
        }
        
        self.experience_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'minimum\s+(\d+)\s+years?',
            r'at least\s+(\d+)\s+years?',
            r'(\d+)-(\d+)\s+years?\s+experience'
        ]
    
    def analyze_job_description(self, jd_text: str) -> List[JobRequirement]:
        """Analyze job description and extract requirements"""
        jd_lower = jd_text.lower()
        requirements = []
        
        # Extract technical skills
        for tech in self.technical_keywords:
            if tech in jd_lower:
                importance = self._determine_importance(jd_text, tech)
                context = self._extract_context(jd_text, tech)
                
                requirements.append(JobRequirement(
                    category="technical",
                    skill=tech,
                    importance=importance,
                    context=context
                ))
        
        # Extract soft skills
        for skill in self.soft_skills:
            if skill in jd_lower:
                importance = self._determine_importance(jd_text, skill)
                context = self._extract_context(jd_text, skill)
                
                requirements.append(JobRequirement(
                    category="soft_skill",
                    skill=skill,
                    importance=importance,
                    context=context
                ))
        
        # Extract experience requirements
        experience_reqs = self._extract_experience_requirements(jd_text)
        requirements.extend(experience_reqs)
        
        # Extract education requirements
        education_reqs = self._extract_education_requirements(jd_text)
        requirements.extend(education_reqs)
        
        return requirements
    
    def _determine_importance(self, jd_text: str, skill: str) -> str:
        """Determine if a skill is required, preferred, or nice to have"""
        skill_context = self._extract_context(jd_text, skill, window=100)
        skill_context_lower = skill_context.lower()
        
        # Required indicators
        required_indicators = ['required', 'must have', 'essential', 'mandatory', 'need']
        if any(indicator in skill_context_lower for indicator in required_indicators):
            return "required"
        
        # Preferred indicators
        preferred_indicators = ['preferred', 'desired', 'ideal', 'bonus', 'plus']
        if any(indicator in skill_context_lower for indicator in preferred_indicators):
            return "preferred"
        
        return "nice_to_have"
    
    def _extract_context(self, text: str, keyword: str, window: int = 200) -> str:
        """Extract context around a keyword"""
        keyword_lower = keyword.lower()
        text_lower = text.lower()
        
        start_pos = text_lower.find(keyword_lower)
        if start_pos == -1:
            return ""
        
        start = max(0, start_pos - window)
        end = min(len(text), start_pos + len(keyword) + window)
        
        return text[start:end].strip()
    
    def _extract_experience_requirements(self, jd_text: str) -> List[JobRequirement]:
        """Extract experience requirements"""
        requirements = []
        
        for pattern in self.experience_patterns:
            matches = re.finditer(pattern, jd_text, re.IGNORECASE)
            for match in matches:
                context = self._extract_context(jd_text, match.group(0))
                requirements.append(JobRequirement(
                    category="experience",
                    skill=f"{match.group(1)} years experience",
                    importance="required",
                    context=context
                ))
        
        return requirements
    
    def _extract_education_requirements(self, jd_text: str) -> List[JobRequirement]:
        """Extract education requirements"""
        education_keywords = [
            'bachelor', 'master', 'phd', 'degree', 'computer science',
            'engineering', 'mathematics', 'certification', 'certified'
        ]
        
        requirements = []
        jd_lower = jd_text.lower()
        
        for edu in education_keywords:
            if edu in jd_lower:
                context = self._extract_context(jd_text, edu)
                requirements.append(JobRequirement(
                    category="education",
                    skill=edu,
                    importance=self._determine_importance(jd_text, edu),
                    context=context
                ))
        
        return requirements

class CVParser:
    """Parses CV content into structured sections"""
    
    def __init__(self):
        self.section_patterns = {
            'contact': r'(?i)(contact|personal\s+information)',
            'summary': r'(?i)(summary|profile|objective|about)',
            'experience': r'(?i)(experience|work\s+history|employment|professional)',
            'education': r'(?i)(education|academic|qualifications)',
            'skills': r'(?i)(skills|technical\s+skills|competencies|technologies)',
            'projects': r'(?i)(projects|portfolio|work\s+samples)',
            'certifications': r'(?i)(certifications|certificates|licenses)',
            'achievements': r'(?i)(achievements|awards|accomplishments|honors)'
        }
    
    def parse_cv(self, cv_text: str) -> List[CVSection]:
        """Parse CV text into structured sections"""
        sections = []
        lines = cv_text.split('\n')
        
        current_section = None
        current_content = []
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line is a section header
            detected_section = self._detect_section(line_stripped)
            
            if detected_section:
                # Save previous section
                if current_section:
                    content = '\n'.join(current_content).strip()
                    if content:
                        sections.append(CVSection(
                            title=current_section,
                            content=content,
                            is_modifiable=self._is_modifiable_section(current_section),
                            keywords=self._extract_keywords(content)
                        ))
                
                # Start new section
                current_section = detected_section
                current_content = []
            else:
                # Add line to current section
                current_content.append(line_stripped)
        
        # Add last section
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                sections.append(CVSection(
                    title=current_section,
                    content=content,
                    is_modifiable=self._is_modifiable_section(current_section),
                    keywords=self._extract_keywords(content)
                ))
        
        return sections
    
    def _detect_section(self, line: str) -> Optional[str]:
        """Detect if a line is a section header"""
        for section_name, pattern in self.section_patterns.items():
            if re.match(pattern, line):
                return section_name
        return None
    
    def _is_modifiable_section(self, section_name: str) -> bool:
        """Determine if a section can be modified/enhanced"""
        modifiable_sections = {'summary', 'skills', 'projects', 'experience'}
        return section_name in modifiable_sections
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from section content"""
        # Simple keyword extraction - can be enhanced with NLP
        content_lower = content.lower()
        keywords = []
        
        # Technical keywords
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws',
            'docker', 'kubernetes', 'git', 'machine learning', 'data science'
        ]
        
        for keyword in tech_keywords:
            if keyword in content_lower:
                keywords.append(keyword)
        
        return keywords

class VectorSearchEngine:
    """Searches the vector store for relevant code examples"""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = "code_embeddings"
        
        # Initialize embedder for query encoding
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def search_for_requirements(self, requirements: List[JobRequirement], 
                               limit_per_requirement: int = 5) -> List[EnhancementSuggestion]:
        """Search vector store for code examples matching job requirements"""
        suggestions = []
        
        for req in requirements:
            if req.category == "technical":
                query_text = f"{req.skill} {req.context}"
                
                # Generate embedding for the requirement
                query_embedding = self._generate_embedding(query_text)
                
                # Search vector store
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding.tolist(),
                    limit=limit_per_requirement,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="technologies",
                                match=MatchValue(value=req.skill)
                            )
                        ]
                    ) if req.skill in ['python', 'javascript', 'java'] else None
                )
                
                # Convert to enhancement suggestions
                logger.info(f"--- Raw Qdrant Payload for skill '{req.skill}' ---")
                for i, result in enumerate(search_results):
                    logger.info(f"Result {i+1} Payload: {result.payload}")
                logger.info("-------------------------------------------------")
                
                for result in search_results:
                    suggestion = EnhancementSuggestion(
                        code_snippet=result.payload.get('content', ''),
                        technologies=result.payload.get('technologies', []),
                        description=self._generate_project_description(
                            result.payload.get('function_name'),
                            result.payload.get('class_name'),
                            result.payload.get('docstring'),
                            result.payload.get('content'),
                            req.skill
                        ),
                        relevance_score=result.score,
                        file_path=result.payload.get('file_path', ''),
                        repo_url=result.payload.get('repo_url', '')
                    )
                    suggestions.append(suggestion)
        
        logger.info(f"Generated {len(suggestions)} suggestions from vector search.")
        for i, sug in enumerate(suggestions[:5]): # Log top 5 suggestions for inspection
            logger.info(f"  Suggestion {i+1}: File={sug.file_path}, Tech={sug.technologies}, Desc={sug.description[:50]}...")
        
        return suggestions

    def _generate_project_description(self, function_name: Optional[str], class_name: Optional[str],
                                       docstring: Optional[str], content: str, skill: str) -> str:
        """Generates a more descriptive and achievement-oriented project entry, avoiding code snippets."""
        
        # Prioritize docstring if it's descriptive and contains quantifiable results
        if docstring and len(docstring.strip()) > 20:
            full_text = docstring.strip()
            quantifiers = re.findall(r'(\d+%|\d+\+?K|\d+\s*(?:times|mins?|hrs?|days?)|reduced|increased|improved|boosted|optimized|achieved|delivered)', full_text, re.IGNORECASE)
            if quantifiers:
                return full_text

        # Use function/class name to generate a more specific description
        if function_name:
            # Clean up function name for better readability
            clean_func_name = function_name.replace('_', ' ').title()
            return f"Engineered '{clean_func_name}' to handle {skill}-related tasks, focusing on performance and scalability."
        if class_name:
            clean_class_name = class_name.replace('_', ' ').title()
            return f"Designed and implemented the '{clean_class_name}' class for robust {skill} functionality."

        # Fallback based on skill, but more achievement-oriented
        if skill:
            return f"Contributed to a project demonstrating practical application of {skill} to solve complex problems."
        
        return "Contributed to a significant project leveraging core technical skills." # Last resort fallback

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for search query"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :]
            
        return embedding.cpu().numpy().flatten()

class CVEnhancer:
    """Enhances CV content based on job requirements and vector search results"""
    
    def __init__(self):
        pass
    
    def enhance_cv_sections(self, cv_sections: List[CVSection], 
                           requirements: List[JobRequirement],
                           suggestions: List[EnhancementSuggestion]) -> List[CVSection]:
        logger.info(f"Inside enhance_cv_sections: suggestions type={type(suggestions)}, len={len(suggestions) if suggestions is not None else 'None'}")
        """Enhance CV sections based on job requirements"""
        enhanced_sections = []
        
        for section in cv_sections:
            if section.is_modifiable:
                enhanced_content = self._enhance_section_content(
                    section, requirements, suggestions
                )
                enhanced_section = CVSection(
                    title=section.title,
                    content=enhanced_content,
                    is_modifiable=section.is_modifiable,
                    keywords=section.keywords
                )
                enhanced_sections.append(enhanced_section)
            else:
                # Keep original content for non-modifiable sections
                enhanced_sections.append(section)
        
        return enhanced_sections
    
    def _enhance_section_content(self, section: CVSection, 
                                requirements: List[JobRequirement],
                                suggestions: List[EnhancementSuggestion]) -> str:
        logger.info(f"Inside _enhance_section_content for section {section.title}: suggestions type={type(suggestions)}, len={len(suggestions) if suggestions is not None else 'None'}")
        """Enhance specific section content"""
        if section.title == 'skills':
            return self._enhance_skills_section(section, requirements, suggestions)
        elif section.title == 'summary':
            return self._enhance_summary_section(section, requirements)
        elif section.title == 'projects':
            return self._enhance_projects_section(section, requirements, suggestions)
        elif section.title == 'experience':
            logger.info(f"Inside _enhance_experience_section call for section {section.title}: suggestions type={type(suggestions)}, len={len(suggestions) if suggestions is not None else 'None'}")
            return self._enhance_experience_section(section, requirements, suggestions)
        else:
            return section.content
    
    def _enhance_skills_section(self, section: CVSection, 
                               requirements: List[JobRequirement],
                               suggestions: List[EnhancementSuggestion]) -> str:
        """Enhance skills section"""
        current_content = section.content
        required_skills = [req.skill for req in requirements if req.category == "technical" and req.importance == "required"]
        
        # Find missing required skills
        missing_skills = []
        for skill in required_skills:
            if skill.lower() not in current_content.lower():
                # Check if we have evidence for this skill in suggestions
                for suggestion in suggestions:
                    if skill in suggestion.technologies:
                        missing_skills.append(skill)
                        break
        
        # Add missing skills
        if missing_skills:
            enhanced_content = current_content + "\n\nAdditional Technical Skills:\n"
            for skill in missing_skills:
                # Find context from suggestions
                context = self._find_skill_context(skill, suggestions)
                enhanced_content += f"• {skill.title()}: {context}\n"
            
            return enhanced_content
        
        return current_content
    
    def _enhance_summary_section(self, section: CVSection, 
                                requirements: List[JobRequirement]) -> str:
        """Enhance summary section"""
        current_content = section.content
        
        # Extract key technologies from requirements
        key_techs = [req.skill for req in requirements 
                    if req.category == "technical" and req.importance == "required"][:5]
        
        if key_techs:
            tech_string = ", ".join(key_techs)
            addition = f"\n\nCore competencies include: {tech_string}."
            return current_content + addition
        
        return current_content
    
    def _enhance_projects_section(self, section: CVSection,
                                 requirements: List[JobRequirement],
                                 suggestions: List[EnhancementSuggestion]) -> str:
        """Enhance projects section by ranking suggestions based on relevance to job requirements."""
        current_content = section.content
        
        # Get unique suggestions to avoid duplicates
        unique_suggestions = list({s.file_path: s for s in suggestions}.values())
        
        # Rank suggestions based on relevance
        required_skills = {req.skill for req in requirements if req.importance == 'required' and req.category == 'technical'}
        
        def rank_suggestion(suggestion):
            matched_skills = [skill for skill in required_skills if skill in suggestion.technologies]
            return len(matched_skills)

        unique_suggestions.sort(key=rank_suggestion, reverse=True)
        
        if unique_suggestions:
            enhanced_content = current_content + "\n\nAdditional Relevant Projects:\n"
            
            for i, suggestion in enumerate(unique_suggestions[:2], 1): # Take top 2 ranked suggestions
                project_name = f"Project {i}: {suggestion.file_path.split('/')[-1].replace('.py', '').title()}"
                tech_list = ", ".join(suggestion.technologies)
                
                # Highlight the skill that matched
                demonstrated_skill = next((skill for skill in required_skills if skill in suggestion.technologies), "key technologies")
                
                # Create a more impactful description
                description = f"Demonstrated expertise in **{demonstrated_skill}** by developing a project that leverages {tech_list} to solve real-world problems. The implementation focuses on clean, scalable code and robust functionality."
                
                enhanced_content += f"\n• {project_name}\n"
                if tech_list:
                    enhanced_content += f"  Technologies: {tech_list}\n"
                enhanced_content += f"  Description: {description}\n"
                enhanced_content += f"  Repository: {suggestion.repo_url}\n"
            
            return enhanced_content
        
        return current_content
    
    def _enhance_experience_section(self, section: CVSection,
                                   requirements: List[JobRequirement],
                                   suggestions: List[EnhancementSuggestion]) -> str:
        """Enhance experience section"""
        # For experience, we'll be more conservative and just add technology mentions
        current_content = section.content
        
        # Find technologies mentioned in suggestions that aren't in current content
        mentioned_techs = []
        for suggestion in suggestions[:3]:
            for tech in suggestion.technologies:
                if tech not in current_content.lower() and tech not in mentioned_techs:
                    mentioned_techs.append(tech)
        
        if mentioned_techs and len(mentioned_techs) <= 3:
            tech_addition = f"\n• Additional experience with: {', '.join(mentioned_techs)}"
            return current_content + tech_addition
        
        return current_content
    
    def _find_skill_context(self, skill: str, suggestions: List[EnhancementSuggestion]) -> str:
        """Find context for a skill from suggestions"""
        for suggestion in suggestions:
            if skill in suggestion.technologies:
                return f"Demonstrated proficiency through practical implementation"
        return "Relevant experience"

class LaTeXGenerator:
    """Generates LaTeX resume from enhanced CV sections"""
    
    def __init__(self):
        self.template = self._load_template()
    
    def generate_latex(self, cv_sections: List[CVSection], 
                      personal_info: Dict[str, str] = None) -> str:
        """Generate LaTeX code from CV sections"""
        
        # Default personal info if not provided
        if not personal_info:
            personal_info = {
                'name': 'Your Name',
                'email': 'your.email@example.com',
                'phone': '+1 (555) 123-4567',
                'linkedin': 'linkedin.com/in/yourprofile',
                'github': 'github.com/yourusername'
            }
        
        # Create sections dictionary
        sections_dict = {section.title: section.content for section in cv_sections}
        
        # Generate LaTeX
        latex_content = self.template.format(
            name=personal_info.get('name', 'Your Name'),
            email=personal_info.get('email', ''),
            phone=personal_info.get('phone', ''),
            linkedin=personal_info.get('linkedin', ''),
            github=personal_info.get('github', ''),
            summary=sections_dict.get('summary', ''),
            experience=self._format_experience(sections_dict.get('experience', '')),
            education=self._format_education(sections_dict.get('education', '')),
            skills=self._format_skills(sections_dict.get('skills', '')),
            projects=self._format_projects(sections_dict.get('projects', '')),
            certifications=sections_dict.get('certifications', ''),
            achievements=sections_dict.get('achievements', '')
        )
        
        return latex_content
    
    def _load_template(self) -> str:
        """Load LaTeX template"""
        return r"""
\documentclass[letterpaper,11pt]{{article}}

\usepackage{{latexsym}}
\usepackage[empty]{{fullpage}}
\usepackage{{titlesec}}
\usepackage{{marvosym}}
\usepackage[usenames,dvipsnames]{{color}}
\usepackage{{verbatim}}
\usepackage{{enumitem}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{fancyhdr}}
\usepackage[english]{{babel}}
\usepackage{{tabularx}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyfoot{{}}
\renewcommand{{\headrulewidth}}{{0pt}}
\renewcommand{{\footrulewidth}}{{0pt}}

\addtolength{{\oddsidemargin}}{{-0.5in}}
\addtolength{{\evensidemargin}}{{-0.5in}}
\addtolength{{\textwidth}}{{1in}}
\addtolength{{\topmargin}}{{-.5in}}
\addtolength{{\textheight}}{{1.0in}}

\urlstyle{{same}}

\raggedbottom
\raggedright
\setlength{{\tabcolsep}}{{0in}}

% Sections formatting
\titleformat{{\section}}{{
  \vspace{{-4pt}}\scshape\raggedright\large
}}{{}}{{0em}}{{}}[\color{{black}}\titlerule \vspace{{-5pt}}]

% Custom commands
\newcommand{{\resumeItem}}[2]{{
  \item\small{{
    \textbf{{#1}}{{: #2 \vspace{{-2pt}}}}
  }}
}}

\newcommand{{\resumeSubheading}}[4]{{
  \vspace{{-1pt}}\item
    \begin{{tabular*}}{{0.97\textwidth}}[t]{{l@{{\extracolsep{{\fill}}}}r}}
      \textbf{{#1}} & #2 \\
      \textit{{\small#3}} & \textit{{\small #4}} \\
    \end{{tabular*}}\vspace{{-5pt}}
}}

\newcommand{{\resumeSubItem}}[2]{{\resumeItem{{#1}}{{#2}}\vspace{{-4pt}}}}

\renewcommand{{\labelitemii}}{{$\circ$}}

\newcommand{{\resumeSubHeadingListStart}}{{\begin{{itemize}}[leftmargin=*]}}
\newcommand{{\resumeSubHeadingListEnd}}{{\end{{itemize}}}}
\newcommand{{\resumeItemListStart}}{{\begin{{itemize}}}}
\newcommand{{\resumeItemListEnd}}{{\end{{itemize}}\vspace{{-5pt}}}}

\begin{{document}}

%----------HEADING-----------------
\begin{{tabular*}}{{\textwidth}}{{l@{{\extracolsep{{\fill}}}}r}}
  \textbf{{\href{{mailto:{email}}}{{\Large {name}}}}} & Email : \href{{mailto:{email}}}{{{email}}}\\
  \href{{https://{linkedin}}}{{{linkedin}}} & Mobile : {phone} \\
  \href{{https://{github}}}{{{github}}} & \\
\end{{tabular*}}

%-----------SUMMARY-----------------
\section{{Professional Summary}}
{summary}

%-----------EXPERIENCE-----------------
\section{{Experience}}
\resumeSubHeadingListStart
{experience}
\resumeSubHeadingListEnd

%-----------EDUCATION-----------------
\section{{Education}}
\resumeSubHeadingListStart
{education}
\resumeSubHeadingListEnd

%-----------SKILLS-----------------
\section{{Technical Skills}}
{skills}

%-----------PROJECTS-----------------
\section{{Projects}}
\resumeSubHeadingListStart
{projects}
\resumeSubHeadingListEnd

%-----------CERTIFICATIONS-----------------
\section{{Certifications}}
{certifications}

%-----------ACHIEVEMENTS-----------------
\section{{Achievements}}
{achievements}

\end{{document}}
"""
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters in a string."""
        text = text.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#').replace('{', '\\{').replace('}', '\\}').replace('~', '\\textasciitilde{}').replace('^', '\\textasciicircum{}').replace('\\', '\\textbackslash{}').replace('$', '\\$')
        return text

    def _format_experience(self, experience_text: str) -> str:
        """Format experience section for LaTeX"""
        if not experience_text:
            return ""

        lines = experience_text.split('\n')
        formatted_entries = []
        current_entry_lines = []

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if stripped_line.startswith('•') or (current_entry_lines and not stripped_line.startswith('•')):
                current_entry_lines.append(stripped_line)
            else:
                if current_entry_lines:
                    formatted_entries.append(self._format_single_job_entry(current_entry_lines))
                current_entry_lines = [stripped_line]
        
        if current_entry_lines:
            formatted_entries.append(self._format_single_job_entry(current_entry_lines))

        return "\n".join(formatted_entries)
    
    def _format_single_job_entry(self, entry_lines: List[str]) -> str:
        """Formats a single job entry (title + bullet points) for LaTeX."""
        if not entry_lines:
            return ""

        # Assume the first line is the job title/company info
        title_line = self._escape_latex(entry_lines[0])
        
        # Placeholder for date range and location/job title
        # In a real scenario, you'd parse these from the CV content
        formatted_entry = f"    \\resumeSubheading{{{title_line}}}{{Date Range}}{{Job Title}}{{Location}}\n"
        
        if len(entry_lines) > 1:
            # The \resumeSubheading already includes an \item. We just need to add sub-items.
            # We remove the extra resumeItemListStart/End and directly use \item for sub-bullet points.
            for item_line in entry_lines[1:]:
                if item_line.strip(): # Ensure line is not empty
                    if item_line.startswith('•'):
                        item_content = item_line[1:].strip()
                    else:
                        item_content = item_line.strip()
                    formatted_entry += f"        \\item {self._escape_latex(item_content)}\n"
            
        return formatted_entry
    
    def _format_education(self, education_text: str) -> str:
        """Format education section for LaTeX"""
        if not education_text:
            return ""
        
        return f"""\\resumeSubheading
      {{University Name}}{{Graduation Date}}
      {{Degree in Major}}{{Location}}"""
    
    def _format_skills(self, skills_text: str) -> str:
        """Format skills section for LaTeX"""
        if not skills_text:
            return ""

        formatted_lines = []
        lines = skills_text.split('\n')
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            
            if ':' in stripped_line and not stripped_line.startswith('•'):
                # Treat as a category: "Category Name: Skill1, Skill2"
                category, skills_str = stripped_line.split(':', 1)
                formatted_lines.append(f"      \\small{{\\textbf{{{self._escape_latex(category.strip())}}}{{:{self._escape_latex(skills_str.strip())}}}}}")
            elif stripped_line.startswith('•'):
                # Treat as a bullet point skill
                formatted_lines.append(f"      \\small{{\\item {self._escape_latex(stripped_line[1:].strip())}}}")
            else:
                # Treat as a general skill item without a specific category prefix
                formatted_lines.append(f"      \\small{{\\item {self._escape_latex(stripped_line)}}}")

        formatted = "\\begin{itemize}[leftmargin=0.15in, label={}]\n"
        formatted += "\n".join(formatted_lines)
        formatted += "\n\\end{itemize}\n"
        
        return formatted
    
    def _format_projects(self, projects_text: str) -> str:
        """Format projects section for LaTeX"""
        if not projects_text:
            return ""

        formatted_projects = []
        lines = projects_text.split('\n')
        
        # Filter out the "Additional Relevant Projects:" header and empty lines
        content_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("Additional Relevant Projects:")]

        current_project_lines = []
        for line in content_lines:
            if line.startswith('• Project'): # New project starts
                if current_project_lines:
                    formatted_projects.append(self._format_single_project_entry(current_project_lines))
                current_project_lines = [line]
            elif current_project_lines: # Add subsequent lines to current project if a project has started
                current_project_lines.append(line)
        
        if current_project_lines: # Add the last project
            formatted_projects.append(self._format_single_project_entry(current_project_lines))

        return "\n".join(formatted_projects)
    
    def _format_single_project_entry(self, project_lines: List[str]) -> str:
        """Formats a single project entry for LaTeX."""
        if not project_lines:
            return ""

        title = ""
        technologies = ""
        description = ""
        repo_url = ""

        # Parse project details from the lines
        for line in project_lines:
            stripped_line = line.strip()
            if stripped_line.startswith('• Project'):
                title = stripped_line.replace('• Project ', '').strip()
            elif stripped_line.startswith('Technologies:'):
                technologies = stripped_line.replace('Technologies:', '').strip()
            elif stripped_line.startswith('Description:'):
                description = stripped_line.replace('Description:', '').strip()
            elif stripped_line.startswith('Repository:'):
                repo_url = stripped_line.replace('Repository:', '').strip()
        
        # Escape LaTeX special characters
        title = self._escape_latex(title)
        technologies = self._escape_latex(technologies)
        description = self._escape_latex(description)
        
        # Format repo_url as a clickable LaTeX link
        formatted_repo_url = f"\\href{{{repo_url}}}{{{repo_url}}}" if repo_url else ""

        # Ensure all 4 arguments for \resumeSubheading are present
        tech_string = f"{{\\small Technologies: {technologies}}}" if technologies else "{}"
        
        # Construct the LaTeX entry
        formatted_entry = f"    \\resumeSubheading{{{title}}}{{}}{{{tech_string}}}{{{formatted_repo_url}}}\\n"
        formatted_entry += "      \\resumeItemListStart\\n"
        formatted_entry += f"        \\item {description}\\n"
        formatted_entry += "      \\resumeItemListEnd\\n"

        return formatted_entry

class CVEnhancementPipeline:
    """Main pipeline that orchestrates the entire CV enhancement process"""
    
    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.document_processor = DocumentProcessor()
        self.jd_analyzer = JobDescriptionAnalyzer()
        self.cv_parser = CVParser()
        self.vector_search = VectorSearchEngine(qdrant_host, qdrant_port)
        self.cv_enhancer = CVEnhancer()
        self.latex_generator = LaTeXGenerator()
    
    def enhance_cv(self, cv_path: str, job_description: str, 
                   output_dir: str = "output") -> Dict[str, Any]:
        """Complete CV enhancement pipeline"""
        
        logger.info("Starting CV enhancement pipeline...")
        
        # Step 1: Extract text from CV
        logger.info("Extracting text from CV...")
        cv_text = self.document_processor.extract_text_from_cv(cv_path)
        
        # Step 2: Analyze job description
        logger.info("Analyzing job description...")
        requirements = self.jd_analyzer.analyze_job_description(job_description)
        
        # Step 3: Parse CV into sections
        logger.info("Parsing CV into sections...")
        cv_sections = self.cv_parser.parse_cv(cv_text)
        
        # Step 4: Search vector store for relevant code examples
        logger.info("Searching for relevant code examples...")
        retrieved_suggestions = self.vector_search.search_for_requirements(requirements)
        logger.info(f"After search_for_requirements: retrieved_suggestions type={type(retrieved_suggestions)}, len={len(retrieved_suggestions) if retrieved_suggestions is not None else 'None'}")

        # Step 5: Enhance CV sections
        logger.info("Enhancing CV sections...")
        logger.info(f"Before enhance_cv_sections: suggestions type={type(retrieved_suggestions)}, len={len(retrieved_suggestions) if retrieved_suggestions is not None else 'None'}")
        enhanced_sections = self.cv_enhancer.enhance_cv_sections(
            cv_sections, requirements, retrieved_suggestions
        )

        # Debug: Print the content of the projects section after enhancement
        for section in enhanced_sections:
            if section.title == 'projects':
                logger.info(f"Enhanced Projects Section Content:\n{section.content}")
                break
        
        # Step 6: Generate LaTeX
        logger.info("Generating LaTeX code...")
        latex_code = self.latex_generator.generate_latex(enhanced_sections)
        
        # Step 7: Save outputs
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LaTeX file
        latex_file = os.path.join(output_dir, "enhanced_resume.tex")
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        # Save analysis report
        report = self._generate_analysis_report(requirements, retrieved_suggestions, enhanced_sections)
        report_file = os.path.join(output_dir, "enhancement_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Compile LaTeX to PDF (if pdflatex is available)
        pdf_file = self._compile_latex_to_pdf(latex_file, output_dir)
        
        result = {
            "status": "success",
            "files": {
                "latex": latex_file,
                "pdf": pdf_file,
                "report": report_file
            },
            "analysis": {
                "requirements_found": len(requirements),
                "suggestions_found": len(retrieved_suggestions),
                "sections_enhanced": len([s for s in enhanced_sections if s.is_modifiable]),
                "top_missing_skills": [req.skill for req in requirements 
                                     if req.importance == "required"][:5]
            }
        }
        
        logger.info("CV enhancement completed successfully!")
        logger.info(f"Files saved to: {output_dir}")
        
        return result
    
    def _generate_analysis_report(self, requirements: List[JobRequirement],
                                 suggestions: List[EnhancementSuggestion],
                                 enhanced_sections: List[CVSection]) -> Dict[str, Any]:
        """Generate detailed analysis report"""
        
        requirements_by_category = {}
        for req in requirements:
            if req.category not in requirements_by_category:
                requirements_by_category[req.category] = []
            requirements_by_category[req.category].append({
                "skill": req.skill,
                "importance": req.importance,
                "context": req.context[:200]  # Truncate for readability
            })
        
        top_suggestions = suggestions[:10]
        suggestions_summary = []
        for suggestion in top_suggestions:
            suggestions_summary.append({
                "technologies": suggestion.technologies,
                "relevance_score": suggestion.relevance_score,
                "file_path": suggestion.file_path,
                "repo_url": suggestion.repo_url
            })
        
        sections_summary = []
        for section in enhanced_sections:
            sections_summary.append({
                "title": section.title,
                "is_modifiable": section.is_modifiable,
                "keywords": section.keywords,
                "content_length": len(section.content),
                "enhanced": section.is_modifiable
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "job_requirements": {
                "total_requirements": len(requirements),
                "by_category": requirements_by_category,
                "required_skills": [req.skill for req in requirements 
                                  if req.importance == "required"],
                "preferred_skills": [req.skill for req in requirements 
                                   if req.importance == "preferred"]
            },
            "vector_search_results": {
                "total_suggestions": len(suggestions),
                "top_matches": suggestions_summary
            },
            "cv_sections": {
                "total_sections": len(enhanced_sections),
                "modifiable_sections": len([s for s in enhanced_sections if s.is_modifiable]),
                "sections_detail": sections_summary
            }
        }
    
    def _compile_latex_to_pdf(self, latex_file: str, output_dir: str) -> Optional[str]:
        """Compile LaTeX to PDF using pdflatex"""
        try:
            import subprocess
            
            # Check if pdflatex is available
            result = subprocess.run(['pdflatex', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning("pdflatex not found. Install TeX Live to generate PDF.")
                return None
            
            # Compile LaTeX to PDF
            pdf_file = latex_file.replace('.tex', '.pdf')
            
            # Run pdflatex twice for proper cross-references
            for _ in range(2):
                result = subprocess.run([
                    'pdflatex', 
                    '-output-directory', output_dir,
                    '-interaction=nonstopmode',
                    latex_file
                ], capture_output=True, text=True, cwd=output_dir)
            
            if os.path.exists(pdf_file):
                logger.info(f"PDF generated successfully: {pdf_file}")
                
                # Clean up auxiliary files
                aux_extensions = ['.aux', '.log', '.out', '.fdb_latexmk', '.fls']
                base_name = os.path.splitext(latex_file)[0]
                for ext in aux_extensions:
                    aux_file = base_name + ext
                    if os.path.exists(aux_file):
                        os.remove(aux_file)
                
                return pdf_file
            else:
                logger.error("PDF generation failed")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return None
                
        except Exception as e:
            logger.error(f"Error compiling LaTeX: {e}")
            return None

# CLI Interface
def main():
    """Command line interface for CV enhancement"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhance CV based on job description using vector search')
    parser.add_argument('cv_path', help='Path to CV file (PDF, DOCX, or TXT)')
    parser.add_argument('job_description', help='Job description text or path to file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--qdrant-host', default='localhost', help='Qdrant host')
    parser.add_argument('--qdrant-port', type=int, default=6333, help='Qdrant port')
    
    args = parser.parse_args()
    
    # Read job description
    if os.path.isfile(args.job_description):
        with open(args.job_description, 'r', encoding='utf-8') as f:
            job_desc = f.read()
    else:
        job_desc = args.job_description
    
    # Initialize pipeline
    pipeline = CVEnhancementPipeline(args.qdrant_host, args.qdrant_port)
    
    try:
        # Run enhancement
        result = pipeline.enhance_cv(args.cv_path, job_desc, args.output)
        
        print("✅ CV Enhancement Completed!")
        print(f"📁 Output directory: {args.output}")
        print(f"📄 LaTeX file: {result['files']['latex']}")
        if result['files']['pdf']:
            print(f"📑 PDF file: {result['files']['pdf']}")
        print(f"📊 Analysis report: {result['files']['report']}")
        
        print(f"\n📈 Analysis Summary:")
        print(f"  • Requirements found: {result['analysis']['requirements_found']}")
        print(f"  • Code suggestions: {result['analysis']['suggestions_found']}")
        print(f"  • Sections enhanced: {result['analysis']['sections_enhanced']}")
        
        if result['analysis']['top_missing_skills']:
            print(f"  • Key skills to highlight: {', '.join(result['analysis']['top_missing_skills'][:3])}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Usage example
if __name__ == "__main__":
    # Example of how to use the pipeline programmatically
    if len(os.sys.argv) > 1:
        main()
    else:
        # Demo usage
        print("CV Enhancement Pipeline Demo")
        print("===========================")
        
        # Example job description
        job_description = """
        We are looking for a Senior Python Developer with 5+ years of experience.
        
        Required Skills:
        - Python programming with Django or Flask
        - RESTful API development
        - Database design (PostgreSQL, MongoDB)
        - Docker and containerization
        - Git version control
        - AWS cloud services
        
        Preferred Skills:
        - Machine Learning experience
        - React.js or Vue.js
        - Microservices architecture
        - CI/CD pipelines
        
        You will be working on scalable web applications and data processing pipelines.
        Strong communication skills and ability to work in agile teams is essential.
        """
        
        # Initialize pipeline
        pipeline = CVEnhancementPipeline()
        
        print("To use this pipeline:")
        print("1. Ensure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        print("2. Make sure you have populated the vector store with GitHub code")
        print("3. Run: python cv_enhancement.py your_cv.pdf job_description.txt")
        print("\nOr use programmatically:")
        print("pipeline.enhance_cv('path/to/cv.pdf', job_description_text)")

# Additional utility functions
def create_sample_job_description():
    """Create a sample job description file for testing"""
    sample_jd = """
Senior Software Engineer - AI/ML Platform

We are seeking an experienced Senior Software Engineer to join our AI/ML platform team. 
You will be responsible for building and maintaining scalable machine learning infrastructure.

Required Qualifications:
- 5+ years of software development experience
- Strong proficiency in Python programming
- Experience with machine learning frameworks (TensorFlow, PyTorch, scikit-learn)
- Knowledge of data processing libraries (pandas, numpy)
- Experience with cloud platforms (AWS, GCP, or Azure)
- Proficiency with containerization (Docker, Kubernetes)
- Strong understanding of RESTful APIs and microservices
- Experience with SQL and NoSQL databases
- Git version control and collaborative development

Preferred Qualifications:
- Experience with MLOps tools and practices
- Knowledge of big data technologies (Spark, Kafka)
- Frontend development experience (React, Vue.js)
- Experience with monitoring and observability tools
- Background in distributed systems
- Familiarity with DevOps practices and CI/CD pipelines

Responsibilities:
- Design and implement scalable ML serving infrastructure
- Develop data processing pipelines for training and inference
- Collaborate with data scientists to productionize models
- Maintain high availability and performance of ML systems
- Implement monitoring and alerting for production systems
- Participate in code reviews and technical discussions

This role offers the opportunity to work with cutting-edge AI/ML technologies
in a fast-paced, collaborative environment.
"""
    
    with open('sample_job_description.txt', 'w', encoding='utf-8') as f:
        f.write(sample_jd)
    
    print("Sample job description created: sample_job_description.txt")