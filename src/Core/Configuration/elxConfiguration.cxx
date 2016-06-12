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
#ifndef __elxConfiguration_CXX__
#define __elxConfiguration_CXX__

#include "elxConfiguration.h"

namespace elastix
{

/**
 * ********************* Constructor ****************************
 */

Configuration::Configuration()
{
  /** Initialize stuff. */
  this->m_ParameterFileName     = "";
  this->m_ParameterFileParser   = ParameterFileParserType::New();
  this->m_ParameterMapInterface = ParameterMapInterfaceType::New();

  this->m_IsInitialized              = false;
  this->m_ElastixLevel               = 0;
  this->m_TotalNumberOfElastixLevels = 1;

} // end Constructor()


/**
 * ******************** PrintParameterFile ***************************
 */

void
Configuration
::PrintParameterFile( void ) const
{
  /** Read what's in the parameter file. */
  std::string params = this->m_ParameterFileParser->ReturnParameterFileAsString();

  /** Separate clearly in log-file. */
  xl::xout[ "logonly" ] << std::endl << "=============== start of ParameterFile: "
                        << this->GetParameterFileName() << " ===============" << std::endl;

  /** Write the parameter file. */
  xl::xout[ "logonly" ] << params;
  //std::cerr << params;

  /** Separate clearly in log-file. */
  xl::xout[ "logonly" ] << std::endl << "=============== end of ParameterFile: "
                        << this->GetParameterFileName() << " ===============\n" << std::endl;

} // end PrintParameterFile()


/**
 * ************************ BeforeAll ***************************
 */

int
Configuration
::BeforeAll( void )
{
#ifndef _ELASTIX_BUILD_LIBRARY
  this->PrintParameterFile();
#endif
  return 0;

} // end BeforeAll()


/**
 * ************************ BeforeAllTransformix ***************************
 */

int
Configuration
::BeforeAllTransformix( void )
{
  this->PrintParameterFile();
  return 0;

} // end BeforeAllTransformix()


/**
 * ********************** Initialize ****************************
 */

int
Configuration
::Initialize( const CommandLineArgumentMapType & _arg )
{
  /** The first part is getting the command line arguments and setting them
   * in the configuration. From the command line arguments we find the name
   * of the parameter text file. The second part is then to get and set the
   * parameter in this configuration.
   */

  /** Store the command line arguments. */
  this->m_CommandLineArgumentMap = _arg;

  /** This function can either be called by elastix or transformix.
   * If called by elastix the command line argument "-p" has to be
   * specified. If called by transformix the command line argument
   * "-tp" has to be specified.
   * NOTE: this implies that one can not use "-tp" for elastix and
   * "-p" for transformix.
   */
  std::string p  = this->GetCommandLineArgument( "-p" );
  std::string tp = this->GetCommandLineArgument( "-tp" );

  if( p != "" && tp == "" )
  {
    /** elastix called Initialize(). */
    this->SetParameterFileName( p.c_str() );
  }
  else if( p == "" && tp != "" )
  {
    /** transformix called Initialize(). */
    this->SetParameterFileName( tp.c_str() );
  }
  else if( p == "" && tp == "" )
  {
    xl::xout[ "error" ] << "ERROR: No (Transform-)Parameter file has been entered" << std::endl;
    xl::xout[ "error" ] << "for elastix: command line option \"-p\"" << std::endl;
    xl::xout[ "error" ] << "for transformix: command line option \"-tp\"" << std::endl;
    return 1;
  }
  else
  {
    /** Both "p" and "tp" are used, which is prohibited. */
    xl::xout[ "error" ] << "ERROR: Both \"-p\" and \"-tp\" are used, "
                        << "which is prohibited." << std::endl;
    return 1;
  }

  /** Read the ParameterFile. */
  this->m_ParameterFileParser->SetParameterFileName( this->m_ParameterFileName );
  try
  {
    xl::xout[ "standard" ] << "Reading the elastix parameters from file ...\n" << std::endl;
    this->m_ParameterFileParser->ReadParameterFile();
  }
  catch( itk::ExceptionObject & excp )
  {
    xl::xout[ "error" ] << "ERROR: when reading the parameter file:\n"
                        << excp << std::endl;
    return 1;
  }

  /** Connect the parameter file reader to the interface. */
  this->m_ParameterMapInterface->SetParameterMap(
    this->m_ParameterFileParser->GetParameterMap() );

  /** Silently check in the parameter file if error messages should be printed. */
  this->m_ParameterMapInterface->SetPrintErrorMessages( false );
  bool printErrorMessages = true;
  this->ReadParameter( printErrorMessages, "PrintErrorMessages", 0, false );
  this->m_ParameterMapInterface->SetPrintErrorMessages( printErrorMessages );

  /** Set the initialized flag. */
  this->m_IsInitialized = true;

  /** Return a value.*/
  return 0;

} // end Initialize()


/**
 * ********************** Initialize ****************************
 */

int
Configuration
::Initialize( const CommandLineArgumentMapType & _arg,
  ParameterFileParserType::ParameterMapType & inputMap )
{
  /** The first part is getting the command line arguments and setting them
   * in the configuration. From the command line arguments we find the name
   * of the parameter text file. The second part is then to get and set the
   * parameter in this configuration.
   */

  /** Store the command line arguments. */
  this->m_CommandLineArgumentMap = _arg;

  this->m_ParameterMapInterface->SetParameterMap( inputMap );

  /** Silently check in the parameter file if error messages should be printed. */
  this->m_ParameterMapInterface->SetPrintErrorMessages( false );
  bool printErrorMessages = true;
  this->ReadParameter( printErrorMessages, "PrintErrorMessages", 0, false );
  this->m_ParameterMapInterface->SetPrintErrorMessages( printErrorMessages );

  /** Set the initialized flag. */
  this->m_IsInitialized = true;

  /** Return a value.*/
  return 0;

} // end Initialize()


/**
 * ********************** IsInitialized ***************************
 */

bool
Configuration
::IsInitialized( void ) const
{
  return this->m_IsInitialized;

} // end IsInitialized()


/**
 * ****************** GetCommandLineArgument ********************
 */

const std::string
Configuration
::GetCommandLineArgument( const std::string & key ) const
{
  /** Check if the argument was given. If no return "". */
  if( this->m_CommandLineArgumentMap.count( key ) == 0 )
  {
    return "";
  }

  return this->m_CommandLineArgumentMap.find( key )->second.c_str();

} // end GetCommandLineArgument()


/**
 * ****************** SetCommandLineArgument ********************
 */

void
Configuration
::SetCommandLineArgument( const std::string & key, const std::string & value )
{
  /** Remove all (!) entries with key 'key' and
   * insert one entry ( key, value ).
   */
  this->m_CommandLineArgumentMap.erase( key );
  this->m_CommandLineArgumentMap.insert( CommandLineEntryType( key, value ) );

} // end SetCommandLineArgument()


} // end namespace elastix

#endif // end #ifndef __elxMyConfiguration_CXX__
