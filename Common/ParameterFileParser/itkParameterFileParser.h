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
#ifndef itkParameterFileParser_h
#define itkParameterFileParser_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkMacro.h"

#include <map>
#include <string>
#include <vector>

namespace itk
{

/** \class ParameterFileParser
 *
 * \brief Implements functionality to read a parameter file.
 *
 * A parameter file is a text file that contains parameters and their values.
 * Parameters should be specified obeying certain rules.\n
 * 1) A single parameter should be on a single line\n
 * 2) A parameter should be specified between brackets: (...)\n
 * 3) Parameters are specified by a single name, followed by one or more
 *      values, all separated by spaces\n
 * 4) Values that are strings should be quoted using "\n
 * 5) Values that are numbers should be unquoted\n
 *
 * For example: \n
 * (ParameterName1 "string1" "string2")\n
 * (ParameterName2 3 5.8)\n
 * (ParameterName3 "true" "false" "true")\n
 *
 * The parameter file is read, and parameter name-value combinations are
 * stored in an std::map< std::string, std::vector<std:string> >, where the
 * string is the parameter name, and the vector of strings are the values.
 * Exceptions are raised in case:\n
 * - the parameter text file cannot be opened,\n
 * - rule 2 or 3 is not satisfied,\n
 * - the parameter name or value contains invalid characters.\n
 *
 * Here is an example on how to use this class:\n
 *
 * itk::ParameterFileParser::Pointer parser = itk::ParameterFileParser::New();
 * parser->SetParameterFileName( parameterFileName );
 * try
 * {
 *   parser->Initialize();
 * }
 * catch ( itk::ExceptionObject & e )
 * {
 *   ...
 * }
 *
 * The resulting map can be accessed via:\n
 *
 * parser->GetParameterMap();
 *
 * \sa itk::ParameterMapInterface
 */

class ParameterFileParser : public Object
{
public:
  /** Standard ITK typedefs. */
  using Self = ParameterFileParser;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ParameterFileParser, Object);

  /** Typedefs. */
  using ParameterValuesType = std::vector<std::string>;
  using ParameterMapType = std::map<std::string, ParameterValuesType>;

  /** Set the name of the file containing the parameters. */
  itkSetStringMacro(ParameterFileName);
  itkGetStringMacro(ParameterFileName);

  /** Return the parameter map. */
  const ParameterMapType &
  GetParameterMap() const;

  /** Read the parameters in the parameter map. */
  void
  ReadParameterFile();

  /** Read the parameter file and return the content as a string.
   * Useful for printing the content.
   */
  std::string
  ReturnParameterFileAsString();

  /** Read the specified file into a parameter map and return the map. */
  static ParameterMapType
  ReadParameterMap(const std::string & fileName);

protected:
  ParameterFileParser();
  ~ParameterFileParser() override;

private:
  ParameterFileParser(const Self &) = delete;
  void
  operator=(const Self &) = delete;

  /** Performs the following checks:
   * - Is a filename is given
   * - Does the file exist
   * - Is a text file, i.e. does it end with .txt
   * If one of these conditions fail, an exception is thrown.
   */
  void
  BasicFileChecking() const;

  /** Checks a line.
   * - Returns  true if it is a valid line: containing a parameter.
   * - Returns false if it is a valid line: empty or comment.
   * - Throws an exception if it is not a valid line.
   */
  bool
  CheckLine(const std::string & line, std::string & lineOut) const;

  /** Fills m_ParameterMap with valid entries. */
  void
  GetParameterFromLine(const std::string & fullLine, const std::string & line);

  /** Splits a line in parameter name and values. */
  void
  SplitLine(const std::string & fullLine, const std::string & line, std::vector<std::string> & splittedLine) const;

  /** Uniform way to throw exceptions when the parameter file appears to be
   * invalid.
   */
  void
  ThrowException(const std::string & line, const std::string & hint) const;

  /** Member variables. */
  std::string      m_ParameterFileName;
  ParameterMapType m_ParameterMap;
};

} // end of namespace itk

#endif // end itkParameterFileParser_h
