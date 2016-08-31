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
#ifndef __itkCommandLineArgumentParser_h
#define __itkCommandLineArgumentParser_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include <vector>
#include <string>
#include <map>
#include <cstdlib>
#include <iostream>

namespace itk
{

/**
 * \class CommandLineArgumentParser
 *
 * \brief This class parses command line arguments. This
 * makes it easy to get the desired argument.
 *
 * Keys are identified as a "-" followed by a string; subsequent
 * entries that are not keys are the values. One or more values can
 * be specified or even no values. For example:
 *
 *   -key1 value1 value2 value3 \n
 *   -key2 value4
 *   -key3
 *
 * Typical use:
 *
 * #include "itkCommandLineArgumentParser.h"
 * int main( int argc, char** argv )
 * {
 *   // Create a command line argument parser.
 *   itk::CommandLineArgumentParser::Pointer parser = itk::CommandLineArgumentParser::New();
 *   parser->SetCommandLineArguments( argc, argv );
 *
 *   std::vector<std::string> arguments1;
 *   bool returnValue1 = parser->GetCommandLineArgument( "-key1", arguments1 );
 *
 *   float argument2;
 *   bool returnValue2 = parser->GetCommandLineArgument( "-key2", argument2 );
 *
 *   bool argumentExists = parser->ArgumentExists( "-key3" );
 * }
 *
 * Arguments can be initialized to default values, which will be left
 * untouched if the key is not provided at the command line. If an
 * argument is initialized with a vector of "size" > 1, and if only
 * one (1) argument is provided in the command line, we create a
 * vector of size "size" and fill it with the single argument.
 *
 * Internally, the command line arguments are stored in an std::map
 * of the argument (key) as an std::string together with the index.
 * We make use of the casting functionality of string streams to
 * automatically cast the stored string to the requested type.
 *
 */

class CommandLineArgumentParser :
  public Object
{
public:

  /** Standard class typedefs. */
  typedef CommandLineArgumentParser  Self;
  typedef Object                     Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  enum ReturnValue { PASSED, FAILED, HELPREQUESTED };

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( CommandLineArgumentParser, Object );

  /** Set the command line arguments in a vector of strings. */
  void SetCommandLineArguments( int argc, char ** argv );

  /** Ensure that all required arguments have been passed. */
  ReturnValue CheckForRequiredArguments() const;

  /** Map to store the arguments and their indices. */
  typedef std::size_t                        IndexType;
  typedef std::map< std::string, IndexType > ArgumentMapType;
  typedef ArgumentMapType::value_type        EntryType;

  /** Copy argv in a map. */
  void CreateArgumentMap( void );

  /** Print all arguments. */
  void PrintAllArguments( void ) const;

  /** Checks if an argument is given. */
  bool ArgumentExists( const std::string & key ) const;

  /** Checks if exactly one of the specified arguments is given. */
  bool ExactlyOneExists( const std::vector< std::string > & keys ) const;

  /** Mark an argument as required. */
  void MarkArgumentAsRequired(
    const std::string & argument, const std::string & helpText = "" );

  /** Mark exactly one of the specified arguments as required. */
  void MarkExactlyOneOfArgumentsAsRequired(
    const std::vector< std::string > & arguments, const std::string & helpText = "" );

  itkSetMacro( ProgramHelpText, std::string );
  itkGetMacro( ProgramHelpText, std::string );

  /** Get command line argument if arg is a vector type. */
  template< class T >
  bool GetCommandLineArgument(
    const std::string & key, std::vector< T > & arg )
  {
    /** Check for the key. */
    IndexType keyIndex, nextKeyIndex;
    keyIndex = nextKeyIndex = 0;
    bool keyFound = this->FindKey( key, keyIndex, nextKeyIndex );
    if( !keyFound ) { return false; }

    /** If a vector of size oldSize > 1 is given to this function, and if
     * only one (1) argument is provided in the command line, we create
     * a vector of size oldSize and fill it with the single argument.
     */
    IndexType oldSize = arg.size();
    if( oldSize > 1 && nextKeyIndex - keyIndex == 2 )
    {
      /** Cast the string to type T. */
      T    casted;
      bool castSuccesful = this->StringCast( this->m_Argv[ keyIndex + 1 ], casted );

      /** Check if the cast was successful. */
      if( !castSuccesful )
      {
        std::stringstream ss;
        ss << "ERROR: Casting entry number " << 0
           << " for the parameter \"" << key
           << "\" failed!\n"
           << "  You tried to cast \"" << this->m_Argv[ keyIndex + 1 ]
           << "\" from std::string to "
           << typeid( arg[ 0 ] ).name() << std::endl;

        itkExceptionMacro( << ss.str() );
      }

      /** Fill the arg vector with the casted value. */
      arg.clear();
      arg.resize( oldSize, casted );

      return true;
    }

    /** Otherwise, gather the arguments and put them in arg. */
    IndexType newSize = nextKeyIndex - keyIndex - 1;
    newSize = newSize > oldSize ? newSize : oldSize;
    arg.resize( newSize );
    IndexType j = 0;
    for( IndexType i = keyIndex + 1; i < nextKeyIndex; i++ )
    {
      /** Cast the string to type T. */
      T    casted;
      bool castSuccesful = this->StringCast( this->m_Argv[ i ], casted );

      /** Check if the cast was successful. */
      if( !castSuccesful )
      {
        std::stringstream ss;
        ss << "ERROR: Casting entry number " << i
           << " for the parameter \"" << key
           << "\" failed!\n"
           << "  You tried to cast \"" << this->m_Argv[ i ]
           << "\" from std::string to "
           << typeid( arg[ j ] ).name() << std::endl;

        itkExceptionMacro( << ss.str() );
      }

      arg[ j ] = casted;
      ++j;
    }
    return true;

  }  // end GetCommandLineArgument()


  /** Get command line argument if arg is not a vector type.
    * We do this by creating a 1D vector, using the GetCommandLineArgument
    * for vector types, and then returning the first element.
    */
  template< class T >
  bool GetCommandLineArgument( const std::string & key, T & arg )
  {
    std::vector< T > vec;
    bool             returnvalue = this->GetCommandLineArgument( key, vec );
    if( returnvalue ) { arg = vec[ 0 ]; }

    return returnvalue;

  }  // end GetCommandLineArgument()


protected:

  CommandLineArgumentParser();
  virtual ~CommandLineArgumentParser() {}

  /** General functionality: find a key. */
  bool FindKey( const std::string & key,
    IndexType & keyIndex, IndexType & nextKeyIndex ) const;

  /** General functionality: Check if key is a number or not. */
  bool IsANumber( const std::string & arg ) const;

  /** A templated function to cast strings to a type T.
   * Returns true when casting was successful and false otherwise.
   * We make use of the casting functionality of string streams.
   */
  template< class T >
  bool StringCast( const std::string & parameterValue, T & casted ) const
  {
    std::stringstream ss( parameterValue );
    ss >> casted;
    if( ss.bad() || ss.fail() )
    {
      return false;
    }
    return true;

  } // end StringCast()


  /** Provide a specialization for std::string, since the general StringCast
   * (especially ss >> casted) will not work for strings containing spaces.
   */
  bool StringCast( const std::string & parameterValue, std::string & casted ) const;

  /** A vector of strings to store the command line arguments. */
  std::vector< std::string > m_Argv;

  /** A map to store the arguments and their indices. The arguments are stored
    * INCLUDING the leading dash. I.e. an example pair is ("-test", 2)
    */
  ArgumentMapType m_ArgumentMap;

  /** The list of required arguments. They are stored with an accompanying help text string. */
  std::vector< std::pair< std::string, std::string > > m_RequiredArguments;

  /** A list of arguments with the condition that exactly one in each set must exist. */
  std::vector< std::pair< std::vector< std::string >, std::string > > m_RequiredExactlyOneArguments;

  std::string m_ProgramHelpText;

private:

  CommandLineArgumentParser( const Self & ); // purposely not implemented
  void operator=( const Self & );            // purposely not implemented

};

// end class CommandLineArgumentParser

} // end namespace itk

#endif // end #ifndef __itkCommandLineArgumentParser_h
