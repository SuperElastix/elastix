/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __itkParameterFileParser_cxx
#define __itkParameterFileParser_cxx

#include "itkParameterFileParser.h"

#include <itksys/SystemTools.hxx>
#include <itksys/RegularExpression.hxx>

#include <fstream>

namespace itk
{

/**
 * **************** Constructor ***************
 */

ParameterFileParser
::ParameterFileParser() = default;


/**
 * **************** Destructor ***************
 */

ParameterFileParser
::~ParameterFileParser() = default;


/**
 * **************** GetParameterMap ***************
 */

const ParameterFileParser::ParameterMapType &
ParameterFileParser
::GetParameterMap( void ) const
{
  return this->m_ParameterMap;

} // end GetParameterMap()


/**
 * **************** ReadParameterFile ***************
 */

void
ParameterFileParser
::ReadParameterFile( void )
{
  /** Perform some basic checks. */
  this->BasicFileChecking();

  /** Open the parameter file for reading. */
  std::ifstream parameterFile( this->m_ParameterFileName );

  /** Check if it opened. */
  if( !parameterFile.is_open() )
  {
    itkExceptionMacro( << "ERROR: could not open "
                       << this->m_ParameterFileName
                       << " for reading." );
  }

  /** Clear the map. */
  this->m_ParameterMap.clear();

  /** Loop over the parameter file, line by line. */
  std::string lineIn;
  std::string lineOut;
  while( parameterFile.good() )
  {
    /** Extract a line. */
    itksys::SystemTools::GetLineFromStream( parameterFile, lineIn );

    /** Check this line. */
    const bool validLine = this->CheckLine( lineIn, lineOut );

    if( validLine )
    {
      /** Get the parameter name from this line and store it. */
      this->GetParameterFromLine( lineIn, lineOut );
    }
    // Otherwise, we simply ignore this line

  }

} // end ReadParameterFile()


/**
 * **************** BasicFileChecking ***************
 */

void
ParameterFileParser
::BasicFileChecking( void ) const
{
  /** Check if the file name is given. */
  if( this->m_ParameterFileName.empty() )
  {
    itkExceptionMacro( << "ERROR: FileName has not been set." );
  }

  /** Basic error checking: existence. */
  const bool exists = itksys::SystemTools::FileExists(
    this->m_ParameterFileName );
  if( !exists )
  {
    itkExceptionMacro( << "ERROR: the file "
                       << this->m_ParameterFileName
                       << " does not exist." );
  }

  /** Basic error checking: file or directory. */
  const bool isDir = itksys::SystemTools::FileIsDirectory(
    this->m_ParameterFileName );
  if( isDir )
  {
    itkExceptionMacro( << "ERROR: the file "
                       << this->m_ParameterFileName
                       << " is a directory." );
  }

  /** Check the extension. */
  const std::string ext = itksys::SystemTools::GetFilenameLastExtension(
    this->m_ParameterFileName );
  if( ext != ".txt" )
  {
    itkExceptionMacro( << "ERROR: the file "
                       << this->m_ParameterFileName
                       << " should be a text file (*.txt)." );
  }

} // end BasicFileChecking()


/**
 * **************** CheckLine ***************
 */

bool
ParameterFileParser
::CheckLine( const std::string & lineIn, std::string & lineOut ) const
{
  /** Preprocessing of lineIn:
   * 1) Replace tabs with spaces
   * 2) Remove everything after comment sign //
   * 3) Remove leading spaces
   * 4) Remove trailing spaces
   */
  lineOut = lineIn;
  itksys::SystemTools::ReplaceString( lineOut, "\t", " " );

  itksys::RegularExpression commentPart( "//" );
  if( commentPart.find( lineOut ) )
  {
    lineOut = lineOut.substr( 0, commentPart.start() );
  }

  itksys::RegularExpression leadingSpaces( "^[ ]*(.*)" );
  leadingSpaces.find( lineOut );
  lineOut = leadingSpaces.match( 1 );

  itksys::RegularExpression trailingSpaces( "[ \t]+$" );
  if( trailingSpaces.find( lineOut ) )
  {
    lineOut = lineOut.substr( 0, trailingSpaces.start() );
  }

  /**
   * Checks:
   * 1. Empty line -> false
   * 2. Comment (line starts with "//") -> false
   * 3. Line is not between brackets (...) -> exception
   * 4. Line contains less than two words -> exception
   *
   * Otherwise return true.
   */

  /** 1. Check for non-empty lines. */
  itksys::RegularExpression reNonEmptyLine( "[^ ]+" );
  const bool                match1 = reNonEmptyLine.find( lineOut );
  if( !match1 )
  {
    return false;
  }

  /** 2. Check for comments. */
  itksys::RegularExpression reComment( "^//" );
  const bool                match2 = reComment.find( lineOut );
  if( match2 )
  {
    return false;
  }

  /** 3. Check if line is between brackets. */
  if( !itksys::SystemTools::StringStartsWith( lineOut, "(" )
    || !itksys::SystemTools::StringEndsWith( lineOut, ")" ) )
  {
    const std::string hint = "Line is not between brackets: \"(...)\".";
    this->ThrowException( lineIn, hint );
  }

  /** Remove brackets. */
  lineOut = lineOut.substr( 1, lineOut.size() - 2 );

  /** 4. Check: the line should contain at least two words. */
  itksys::RegularExpression reTwoWords( "([ ]+)([^ ]+)" );
  const bool                match4 = reTwoWords.find( lineOut );
  if( !match4 )
  {
    const std::string hint = "Line does not contain a parameter name and value.";
    this->ThrowException( lineIn, hint );
  }

  /** At this point we know its at least a line containing a parameter.
   * However, this line can still be invalid, for example:
   * (string &^%^*)
   * This will be checked later.
   */

  return true;

} // end CheckLine()


/**
 * **************** GetParameterFromLine ***************
 */

void
ParameterFileParser
::GetParameterFromLine( const std::string & fullLine,
  const std::string & line )
{
  /** A line has a parameter name followed by one or more parameters.
   * They are all separated by one or more spaces (all tabs have been
   * removed previously) or by quotes in case of strings. So,
   * 1) we split the line at the spaces or quotes
   * 2) the first one is the parameter name
   * 3) the other strings that are not a series of spaces, are parameter values
   */

  /** 1) Split the line. */
  std::vector< std::string > splittedLine;
  this->SplitLine( fullLine, line, splittedLine );

  /** 2) Get the parameter name. */
  std::string parameterName = splittedLine[ 0 ];
  itksys::SystemTools::ReplaceString( parameterName, " ", "" );
  splittedLine.erase( splittedLine.begin() );

  /** 3) Get the parameter values. */
  std::vector< std::string > parameterValues;
  for( const auto& value: splittedLine )
  {
    if( ! value.empty() )
    {
      parameterValues.push_back( value );
    }
  }

  /** 4) Perform some checks on the parameter name. */
  itksys::RegularExpression reInvalidCharacters1( "[.,:;!@#$%^&-+|<>?]" );
  const bool                match = reInvalidCharacters1.find( parameterName );
  if( match )
  {
    const std::string hint = "The parameter \""
      + parameterName
      + "\" contains invalid characters (.,:;!@#$%^&-+|<>?).";
    this->ThrowException( fullLine, hint );
  }

  /** 5) Perform checks on the parameter values. */
  itksys::RegularExpression reInvalidCharacters2( "[,;!@#$%&|<>?]" );
  for( const auto& parameterValue: parameterValues )
  {
    /** For all entries some characters are not allowed. */
    if( reInvalidCharacters2.find( parameterValue ) )
    {
      const std::string hint = "The parameter value \""
        + parameterValue
        + "\" contains invalid characters (,;!@#$%&|<>?).";
      this->ThrowException( fullLine, hint );
    }
  }

  /** 6) Insert this combination in the parameter map. */
  if( this->m_ParameterMap.count( parameterName ) )
  {
    const std::string hint = "The parameter \""
      + parameterName
      + "\" is specified more than once.";
    this->ThrowException( fullLine, hint );
  }
  else
  {
    this->m_ParameterMap.insert( make_pair( parameterName, parameterValues ) );
  }

} // end GetParameterFromLine()


/**
 * **************** SplitLine ***************
 */

void
ParameterFileParser
::SplitLine( const std::string & fullLine, const std::string & line,
  std::vector< std::string > & splittedLine ) const
{
  splittedLine.clear();
  splittedLine.resize( 1 );

  /** Count the number of quotes in the line. If it is an odd value, the
   * line contains an error; strings should start and end with a quote, so
   * the total number of quotes is even.
   */
  std::size_t numQuotes = itksys::SystemTools::CountChar( line.c_str(), '"' );
  if( numQuotes % 2 == 1 )
  {
    /** An invalid parameter line. */
    const std::string hint = "This line has an odd number of quotes (\").";
    this->ThrowException( fullLine, hint );
  }

  /** Loop over the line. */
  unsigned int                index = 0;
  numQuotes = 0;
  for( const char currentChar: line )
  {
    if( currentChar == '"' )
    {
      /** Start a new element. */
      splittedLine.push_back( "" );
      index++;
      numQuotes++;
    }
    else if( currentChar == ' ' )
    {
      /** Only start a new element if it is not a quote, otherwise just add
       * the space to the string.
       */
      if( numQuotes % 2 == 0 )
      {
        splittedLine.push_back( "" );
        index++;
      }
      else
      {
        splittedLine[ index ].push_back( currentChar );
      }
    }
    else
    {
      /** Add this character to the element. */
      splittedLine[ index ].push_back( currentChar );
    }
  }

} // end SplitLine()


/**
 * **************** ThrowException ***************
 */

void
ParameterFileParser
::ThrowException( const std::string & line, const std::string & hint ) const
{
  /** Construct an error message. */
  const std::string errorMessage
    = "ERROR: the following line in your parameter file is invalid: \n\""
    + line
    + "\"\n"
    + hint
    + "\nPlease correct you parameter file!";

  /** Throw exception. */
  itkExceptionMacro( << errorMessage );

} // end ThrowException()


/**
 * **************** ReturnParameterFileAsString ***************
 */

std::string
ParameterFileParser
::ReturnParameterFileAsString( void )
{
  /** Perform some basic checks. */
  this->BasicFileChecking();

  /** Open the parameter file for reading. */
  std::ifstream parameterFile( this->m_ParameterFileName );

  /** Check if it opened. */
  if( !parameterFile.is_open() )
  {
    itkExceptionMacro( << "ERROR: could not open "
                       << this->m_ParameterFileName
                       << " for reading." );
  }

  /** Loop over the parameter file, line by line. */
  std::string line;
  std::string output;
  while( parameterFile.good() )
  {
    /** Extract a line. */
    itksys::SystemTools::GetLineFromStream( parameterFile, line ); // \todo: returns bool

    output += line + "\n";
  }

  /** Return the string. */
  return output;

} // end ReturnParameterFileAsString()


} // end namespace itk

#endif // end __itkParameterFileParser_cxx
