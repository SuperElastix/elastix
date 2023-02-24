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
#ifndef elxlog_h
#define elxlog_h

#include <string>
#include <sstream>

namespace elastix
{
/**
 * The log class provides a minimal interface to spdlog for elastix itself. "spdlog" is C++ logging library by Gabi
 * Melman, from https://github.com/gabime/spdlog
 */
class log
{
public:
  enum class level : uint8_t
  {
    info,
    warn,
    err,
    off
  };

  /** Does setup the logging system. Optionally enables log to file and/or log to stdout. */
  static bool
  setup(const std::string & log_filename,
        const bool          do_log_to_file,
        const bool          do_log_to_stdout,
        const level         log_level = {});

  /** Does setup and reset the logging system, according to the C++ "RAII" principle) */
  class guard
  {
  public:
    /** Default-constructor, just creates a `guard` object, to be destructed later. */
    guard();

    /** Does setup the logging system. */
    guard(const std::string & log_filename,
          const bool          do_log_to_file,
          const bool          do_log_to_stdout,
          const level         log_level);

    /** Does reset the logging system. */
    ~guard();
  };

  ///@{
  /** Just passes the message to the corresponding spdlog function. */
  static void
  info(const std::string & message);
  static void
  warn(const std::string & message);
  static void
  error(const std::string & message);
  ///@}

  ///@{
  /** Retrieves the string from the specified stream and passes it the corresponding spdlog function. */
  static void
  info(const std::ostream & stream);
  static void
  warn(const std::ostream & stream);
  static void
  error(const std::ostream & stream);
  ///@}

  ///@{
  /** Passes the message only to the log file, not to stdout. */
  static void
  info_to_log_file(const std::string & message);

  static void
  info_to_log_file(const std::ostream & stream);
  ///@}

  ///@{
  /** Passes the message only to stdout, not to the log file. */
  static void
  to_stdout(const std::string & message);

  static void
  to_stdout(const std::ostream & stream);
  ///@}
};

} // end namespace elastix

#endif
