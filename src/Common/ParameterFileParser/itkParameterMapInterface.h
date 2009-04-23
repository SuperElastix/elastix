#ifndef __itkParameterMapInterface_h
#define __itkParameterMapInterface_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkMacro.h"

#include "itkParameterFileParser.h"

#include <iostream>


namespace itk
{
  
/** \class ParameterMapInterface
 *
 * \brief Implements functionality to get parameters from a parameter map.
 *
 * This class requires an std::map of parameter names and values, specified
 * as strings. Such a map can be created by the related class
 * itk::ParameterFileParser. This class implements functionality to get
 * parameters from this map and return them in the desired type. The function
 * that takes care of that is:\n
 *   ReadParameter( T parameterValue, std::string parameterName, ... )\n
 * which is templated over T. For convenience, several flavors of
 * ReadParameter() exist.
 *
 * The layout of ReadParameter is specified below.
 *
 * Warnings are created if the following two conditions are both satisfied:
 * 1) ReadParameter() is called with the function argument printWarningToStream
 *    set to true.
 * 2) The global member variable m_PrintErrorMessages is true.
 *
 * This class can be used in the following way:\n
 *
 * itk::ParameterMapInterface::Pointer p_interface = itk::ParameterMapInterface::New();
 * p_interface->SetParameterMap( parser->GetParameterMap() );
 * p_interface->PrintErrorMessages( true );
 * unsigned long parameterValue = 3;
 * unsigned int index = 2;
 * bool printWarning = true;
 * std::string errorMessage = "";
 * bool success = p_interface->ReadParameter( parameterValue,
 *   "ParameterName", index, printWarning, errorMessage );
 *
 * 
 * Note that some of the templated functions are defined in the header to
 * get it compiling on some platforms.
 *
 * \sa itk::ParameterFileParser
 */

class ParameterMapInterface : public Object
{
public:
 
  /** Standard ITK typedefs. */
  typedef ParameterMapInterface       Self;
  typedef Object                      Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;
  
  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ParameterMapInterface, Object );

  /** Typedefs. */
  typedef ParameterFileParser::ParameterValuesType  ParameterValuesType;
  typedef ParameterFileParser::ParameterMapType     ParameterMapType;

  /** Set the parameter map. */
  void SetParameterMap( const ParameterMapType & parMap );

  /** Option to print error and warning messages to a stream.
   * The default is true. If set to false no messages are printed.
   */
  // \todo: we could think of a warning level. (maybe you want warnings, but
  // not when for example a parameter is not found at entry entry_nr but at entry 0 instead
  itkSetMacro( PrintErrorMessages, bool );
  itkGetConstMacro( PrintErrorMessages, bool );

  /** Get the number of entries for a given parameter. */
  unsigned int CountNumberOfParameterEntries(
    const std::string & parameterName ) const;

  /** Get the desired parameter from the parameter map as type T.
   *
   * When requesting to read a parameter, multiple scenarios exist:
   * 1) The parameter is not found at all
   * 2) The parameter is found, but index entry_nr does not exist
   * 3) The parameter is found at the requested index, and cast is correct
   * 4) The parameter is found at the requested index, but the cast fails
   * What to return for these three options?
   * 1) -> return false + warning if desired
   * 2) -> return false + other warning if desired
   * 3) -> return true and no warning
   * 4) -> Throw exception: there is an error in the parameter file
   *
   */
  template <class T>
  bool ReadParameter( T & parameterValue,
    const std::string & parameterName,
    const unsigned int entry_nr,
    const bool printThisErrorMessage,
    std::string & errorMessage ) const
  {
    /** Reset the error message. */
    errorMessage = "";

    /** Get the number of entries. */
    unsigned int numberOfEntries = this->CountNumberOfParameterEntries(
      parameterName );

    /** Check if the requested parameter exists. */
    if ( numberOfEntries == 0 )
    {
      std::stringstream ss;
      ss << "WARNING: The parameter \"" << parameterName
        << "\", requested at entry number " << entry_nr
        << ", does not exist at all.\n"
        << "  The default value \"" << parameterValue
        << "\" is used instead." << std::endl;
      if ( printThisErrorMessage && this->m_PrintErrorMessages )
      {
        errorMessage = ss.str();
      }

      return false;
    }

    /** Get the vector of parameters. */
    ParameterValuesType vec = this->m_ParameterMap.find( parameterName )->second;

    /** Check if it exists at the requested entry number. */
    if ( entry_nr >= numberOfEntries )
    {
      std::stringstream ss;
      ss << "WARNING: The parameter \"" << parameterName
        << "\" does not exist at entry number " << entry_nr
        << ".\n  The default value \"" << parameterValue
        << "\" is used instead." << std::endl;
      if ( printThisErrorMessage && this->m_PrintErrorMessages )
      {
        errorMessage = ss.str();
      }
      return false;
    }
      
    /** Cast the string to type T. */
    bool castSuccesful = this->StringCast<T>( vec[ entry_nr ], parameterValue );

    /** Check if the cast was successful. */
    if ( !castSuccesful )
    {
      std::stringstream ss;
      ss << "ERROR: Casting entry number " << entry_nr
        << " for the parameter \"" << parameterName
        << "\" failed!\n"
        << "  You tried to cast \"" << vec[ entry_nr ]
        << "\" from std::string to "
        << typeid( parameterValue ).name() << std::endl;

      itkExceptionMacro( << ss.str() );
    }

    return true;

  } // end ReadParameter()

  /** Boolean support. */
  bool ReadParameter( bool & parameterValue,
    const std::string & parameterName,
    const unsigned int entry_nr,
    const bool printThisErrorMessage,
    std::string & errorMessage ) const;

  /** A shorter version of ReadParameter() that does not require the boolean
   * printThisErrorMessage. Instead the default value true is used.
   */
  template <class T>
  bool ReadParameter( T & parameterValue,
    const std::string & parameterName,
    const unsigned int entry_nr,
    std::string & errorMessage ) const
  {
    return this->ReadParameter( parameterValue, parameterName, entry_nr,
      true, errorMessage );
  }

  /** An extended version of ReadParameter() that takes prefixes and
   * default entry numbers (for convenience).
   * This function tries to read parameterName, but also prefix+parameterName.
   * Also, multiple entries are tried, entry_nr as well as default_entry_nr.
   */
  template <class T>
  bool ReadParameter( T & parameterValue,
    const std::string & parameterName,
    const std::string & prefix,
    const unsigned int entry_nr,
    const int default_entry_nr,
    const bool printThisErrorMessage,
    std::string & errorMessage ) const
  {
    std::string fullname = prefix + parameterName;
    bool found = false;

    /** Silently try to read the parameter. */
    std::string dummyString = "";
    if ( default_entry_nr >= 0 )
    {
      /** Try the default_entry_nr if the entry_nr is not found. */
      unsigned int uintdefault = static_cast<unsigned int>( default_entry_nr );
      found |= this->ReadParameter( parameterValue, parameterName, uintdefault,
        false, dummyString );
      found |= this->ReadParameter( parameterValue, parameterName, entry_nr,
        false, dummyString );
      found |= this->ReadParameter( parameterValue, fullname, uintdefault,
        false, dummyString );
      found |= this->ReadParameter( parameterValue, fullname, entry_nr,
        false, dummyString );
    }
    else
    {
      /** Just try the entry_nr. */
      found |= this->ReadParameter( parameterValue, parameterName, entry_nr,
        false, dummyString );
      found |= this->ReadParameter( parameterValue, fullname, entry_nr,
        false, dummyString );
    }

    /** If we haven't found anything, give a warning that the default value
     * provided by the caller is used.
     */
    if ( !found && printThisErrorMessage && this->m_PrintErrorMessages )
    {
      return this->ReadParameter( parameterValue, parameterName, entry_nr,
        true, errorMessage );
    }

    return found;

  }

  /** A shorter version of the extended ReadParameter() that does not require
   * the boolean printThisErrorMessage. Instead the default value true is used.
   */
  template <class T>
  bool ReadParameter( T & parameterValue,
    const std::string & parameterName,
    const std::string & prefix,
    const unsigned int entry_nr,
    const unsigned int default_entry_nr,
    std::string & errorMessage ) const
  {
    return this->ReadParameter( parameterValue, parameterName, prefix,
      entry_nr, default_entry_nr, true, errorMessage );
  }

protected:
  ParameterMapInterface();
  virtual ~ParameterMapInterface();

private:
  ParameterMapInterface(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** Member variable to store the parameters. */
  ParameterMapType  m_ParameterMap;

  bool              m_PrintErrorMessages;
  //OutputStreamType  m_ErrorMessagesOutputStream;

  /** A templated function to cast strings to a type T.
   * Returns true when casting was successful and false otherwise.
   * We make use of the casting functionality of string streams.
   */
  template <class T>
  bool StringCast( const std::string & parameterValue, T & casted ) const
  {
    std::stringstream ss( parameterValue );
    ss >> casted;
    if ( ss.bad() || ss.fail() )
    {
      return false;
    }
    return true;

  } // end StringCast()

}; // end class ParameterMapInterface

} // end of namespace itk

#endif // end __itkParameterMapInterface_h
